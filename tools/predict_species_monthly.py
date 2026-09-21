"""Predict species month by month using all or seasonally active references."""

from __future__ import annotations

import argparse
import calendar
from collections import Counter
import glob
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def load_monthly_species_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    months = list(calendar.month_name)[1:]
    if not isinstance(data, dict) or set(data) != set(months):
        raise ValueError("Species JSON must contain all twelve full English month names.")
    for name, species in data.items():
        if not isinstance(species, list) or any(not isinstance(s, str) or not s.strip() for s in species):
            raise ValueError(f"{name}: expected a list of nonempty species names.")
    return {month: sorted(set(data[name])) for month, name in enumerate(months, 1)}


def resolve_test_files(inputs):
    files = []
    for value in inputs:
        path = Path(value)
        matches = sorted(path.glob("*.npz")) if path.is_dir() else [Path(p) for p in sorted(glob.glob(value))]
        if not matches:
            raise FileNotFoundError(f"No representation files matched {value!r}.")
        for match in matches:
            if not match.is_file() or match.suffix.lower() != ".npz":
                raise ValueError(f"Expected an NPZ file: {match}")
            resolved = str(match.resolve())
            if resolved not in files:
                files.append(resolved)
    return files


def load_split_labels(train_path, test_path):
    shared = Path(train_path).resolve() == Path(test_path).resolve()
    train = pd.read_csv(train_path)
    test = train.copy() if shared else pd.read_csv(test_path)
    if shared and "split" not in train:
        raise ValueError("A shared labels CSV requires explicit train/test splits.")
    def select(labels, split):
        if "split" in labels:
            return labels.loc[labels["split"].eq(split)].copy()
        return labels.assign(split=split)
    return select(train, "train"), select(test, "test")


def add_capture_dates(labels):
    result = labels.copy()
    dates = result["event_id"].astype("string").str.extract(r"_(\d{4}-\d{2}-\d{2})_", expand=False)
    result["capture_date"] = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    if result["capture_date"].isna().any():
        raise ValueError("Some test event IDs contain no _YYYY-MM-DD_ capture date.")
    return result


def select_month_labels(labels, month, year):
    return labels.loc[labels["capture_date"].dt.month.eq(month) & labels["capture_date"].dt.year.eq(year)].copy()


def load_label_rows(representation_file, labels):
    """Load only image-0 embeddings, matching the preceding filter pipeline.

    Read selected rows from the NPZ stream without materializing the full matrix.
    The NPZ must contain files1 and a numeric, C-ordered emb1 matrix.
    """
    labels = labels.copy()
    if "image_nr" in labels:
        labels = labels.loc[labels["image_nr"].eq(0)].copy()
    with np.load(representation_file, allow_pickle=False) as data:
        paths = data["files1"].astype(str)
    lookup = pd.DataFrame({"rec_path": paths, "_embedding_row": np.arange(len(paths))})
    rows = lookup.merge(labels, on="rec_path", how="inner")
    if rows.empty:
        rows["emb"] = pd.Series(dtype=object)
        return rows.drop(columns="_embedding_row")
    indices = np.unique(rows["_embedding_row"].to_numpy())
    embeddings = {}
    with zipfile.ZipFile(representation_file) as archive, archive.open("emb1.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported embedding NPY version: {version}")
        if fortran or len(shape) != 2 or dtype.kind not in "fiu":
            raise ValueError("emb1 must be a C-ordered numeric matrix.")
        if shape[0] != len(paths):
            raise ValueError("files1 and emb1 have different row counts.")
        origin = stream.tell()
        row_bytes = shape[1] * dtype.itemsize
        for index in indices:
            stream.seek(origin + int(index) * row_bytes)
            payload = stream.read(row_bytes)
            if len(payload) != row_bytes:
                raise ValueError("Incomplete embedding row in NPZ.")
            embeddings[index] = np.frombuffer(payload, dtype=dtype).copy()
    rows["emb"] = [embeddings[i] for i in rows["_embedding_row"]]
    sort = [c for c in ("species", "event_id", "image_nr") if c in rows]
    return rows.drop(columns="_embedding_row").sort_values(sort).reset_index(drop=True)


def sample_training_rows(rows, max_per_species, random_state):
    rows = rows.loc[rows["split"].eq("train")].reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    selected = []
    for species in np.unique(rows["species"]):
        indices = np.flatnonzero(rows["species"].to_numpy() == species)
        rng.shuffle(indices)
        selected.extend(indices[:max_per_species])
    rng.shuffle(selected)
    return rows.iloc[selected].reset_index(drop=True)


class SpeciesKNN:
    """Uniform neighbour voting, excluding references from the query's event.

    Vote ties choose the species whose closest neighbour is most similar.
    Equal similarities retain reference order. If fewer than k eligible
    references exist, use all eligible references. Never vote on masked rows.
    """

    def __init__(self, references, *, n_neighbors=10):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")
        self.k = n_neighbors
        self.species = references["species"].to_numpy(dtype=str)
        self.events = references["event_id"].to_numpy(dtype=str)
        self.embeddings = (
            normalize(np.vstack(references["emb"]), norm="l2")
            if len(references) else None
        )

    def predict(self, rows, *, batch_size=1000):
        """Return metadata and predictions; absent ground truth is allowed."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        metadata = [c for c in ("event_id", "rec_path", "image_nr", "capture_date",
                                "dataset_id", "split", "species") if c in rows]
        result = rows[metadata].reset_index(drop=True).rename(columns={"species": "species_gt"})
        result["predicted_species"] = pd.Series([None] * len(rows), dtype=object)
        result["neighbors_used"] = 0
        result["vote_fraction"] = np.nan
        result["nearest_similarity"] = np.nan
        result["prediction_status"] = "no_references"
        if rows.empty or self.embeddings is None:
            return result
        for start in range(0, len(rows), batch_size):
            batch = rows.iloc[start:start + batch_size]
            query = normalize(np.vstack(batch["emb"]), norm="l2")
            similarities = query @ self.embeddings.T
            for offset, event in enumerate(batch["event_id"].astype(str)):
                index = start + offset
                eligible = np.flatnonzero(self.events != event)
                if not len(eligible):
                    result.at[index, "prediction_status"] = "no_eligible_references"
                    continue
                scores = similarities[offset, eligible]
                nearest = eligible[np.argsort(-scores, kind="stable")[:self.k]]
                species, votes = Counter(self.species[nearest]).most_common(1)[0]
                result.at[index, "predicted_species"] = species
                result.at[index, "neighbors_used"] = len(nearest)
                result.at[index, "vote_fraction"] = votes / len(nearest)
                result.at[index, "nearest_similarity"] = float(similarities[offset, nearest[0]])
                result.at[index, "prediction_status"] = "predicted"
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=positive_int, required=True)
    parser.add_argument("--train-representations", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--test-representations", nargs="+", required=True)
    parser.add_argument("--test-labels", required=True)
    parser.add_argument("--reference-mode", choices=("all", "active"), required=True,
                        help="Use all training species or only exact names in the month's JSON list.")
    parser.add_argument("--monthly-species-json",
                        help="Required for active mode; no default. Use the active-known export if desired.")
    parser.add_argument("--known-predictions-csv", required=True,
                        help="Prior predictions CSV: only events with isknown_pred == 1 are predicted.")
    parser.add_argument("--output-dir", required=True, help="New directory for CSVs and summary.json.")
    parser.add_argument("--max-train-per-species", type=positive_int, default=80)
    parser.add_argument("--n-neighbors", type=positive_int, default=10)
    parser.add_argument("--prediction-batch-size", type=positive_int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    if args.reference_mode == "active" and not args.monthly_species_json:
        parser.error("--monthly-species-json is required when --reference-mode is active")
    return args


def select_known_events(labels, prediction_path):
    """Match event IDs; exclude events with any unknown prediction."""
    predictions = pd.read_csv(prediction_path, dtype={"event_id": str})
    if not {"event_id", "isknown_pred"}.issubset(predictions.columns):
        raise ValueError("Known predictions CSV requires event_id and isknown_pred columns.")
    if predictions["event_id"].isna().any() or not predictions["isknown_pred"].isin([0, 1]).all():
        raise ValueError("Known predictions CSV requires nonmissing event IDs and binary predictions.")
    known = set(predictions.loc[predictions["isknown_pred"].eq(1), "event_id"])
    unknown = set(predictions.loc[predictions["isknown_pred"].eq(0), "event_id"])
    return labels.loc[labels["event_id"].astype(str).isin(known - unknown)].copy()


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    seasonal = load_monthly_species_json(args.monthly_species_json) if args.reference_mode == "active" else None
    test_files = resolve_test_files(args.test_representations)
    train, test = load_split_labels(args.train_labels, args.test_labels)
    for name, labels in (("Training", train), ("Test", test)):
        required = {"event_id", "rec_path"} | ({"species"} if name == "Training" else set())
        missing = required - set(labels.columns)
        if missing:
            raise ValueError(f"{name} labels missing columns: {sorted(missing)}")
        if labels["event_id"].isna().any():
            raise ValueError(f"{name} labels contain missing event IDs.")
    if train["species"].isna().any() or train["species"].astype(str).str.strip().eq("").any():
        raise ValueError("Training rows must have species labels.")
    dated = add_capture_dates(test)
    dated = dated.loc[dated["capture_date"].dt.year.eq(args.year)].copy()
    if dated.empty:
        raise ValueError(f"No test labels for year {args.year}.")
    if args.known_predictions_csv:
        dated = select_known_events(dated, args.known_predictions_csv)

    print("Loading and sampling Poleno references once", flush=True)
    references = sample_training_rows(
        load_label_rows(args.train_representations, train),
        args.max_train_per_species, args.random_state,
    )
    if references.empty:
        raise ValueError("No matching training references.")
    available = sorted(references["species"].unique())
    all_model = SpeciesKNN(references, n_neighbors=args.n_neighbors) if seasonal is None else None
    output.mkdir(parents=True, exist_ok=False)
    combined = output / "species_predictions_all_months.csv"
    schema = SpeciesKNN(references.iloc[:0]).predict(dated.iloc[:0])
    for column in ("scenario_year", "scenario_month", "reference_mode", "representation_file"):
        schema[column] = pd.Series(dtype=object)
    schema.to_csv(combined, index=False)
    summary = {"config": vars(args), "reference_rows": len(references),
               "test_files": test_files, "months": []}

    def save_summary():
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save_summary()
    for month in range(1, 13):
        name = calendar.month_name[month]
        configured = seasonal[month] if seasonal is not None else available
        month_references = references.loc[references["species"].isin(configured)]
        model = all_model if all_model is not None else SpeciesKNN(month_references, n_neighbors=args.n_neighbors)
        selected = select_month_labels(dated, month, args.year)
        monthly_file = output / f"{args.year}_{month:02d}_{name}.csv"
        schema.to_csv(monthly_file, index=False)
        report = {"month": month, "month_name": name, "configured_species": configured,
                  "reference_species": sorted(month_references["species"].unique()),
                  "missing_reference_species": sorted(set(configured) - set(available)),
                  "reference_rows": len(month_references), "selected_label_rows": len(selected),
                  "output_rows": 0, "predicted_rows": 0, "unclassified_rows": 0,
                  "evaluated_rows": 0, "accuracy": None, "status": "running"}
        summary["months"].append(report)
        save_summary()
        correct = 0
        if not selected.empty:
            for test_file in test_files:
                print(f"{name}: {len(month_references)} references; loading {test_file}", flush=True)
                rows = load_label_rows(test_file, selected)
                result = model.predict(rows, batch_size=args.prediction_batch_size)
                result["scenario_year"] = args.year
                result["scenario_month"] = month
                result["reference_mode"] = args.reference_mode
                result["representation_file"] = test_file
                result = result.reindex(columns=schema.columns)
                result.to_csv(monthly_file, mode="a", header=False, index=False)
                result.to_csv(combined, mode="a", header=False, index=False)
                predicted = result["prediction_status"].eq("predicted")
                report["output_rows"] += len(result)
                report["predicted_rows"] += int(predicted.sum())
                if "species_gt" in result:
                    labeled = result["species_gt"].notna() & result["species_gt"].astype("string").str.strip().ne("") & predicted
                    report["evaluated_rows"] += int(labeled.sum())
                    correct += int(result.loc[labeled, "species_gt"].eq(result.loc[labeled, "predicted_species"]).sum())
        report["unclassified_rows"] = report["output_rows"] - report["predicted_rows"]
        report["accuracy"] = correct / report["evaluated_rows"] if report["evaluated_rows"] else None
        report["status"] = (
            "no_test_labels" if selected.empty else
            "no_matching_representations" if not report["output_rows"] else
            "no_references" if month_references.empty else
            "completed_with_unclassified_rows" if report["unclassified_rows"] else "completed"
        )
        save_summary()
        print(f"{name}: {report['predicted_rows']} species predictions, {report['status']}", flush=True)
    print(f"Saved species predictions to {output}", flush=True)
    return summary


if __name__ == "__main__":
    run(parse_args())
