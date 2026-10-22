import numpy as np
import pandas as pd


def limit_events_per_species(
    df: pd.DataFrame,
    species_col="species",
    event_col="event_id",
    max_events=50,
    seed=42,
):
    rng = np.random.default_rng(seed)

    selected_events = []

    for species, g in df.groupby(species_col):
        events = g[event_col].unique()

        if len(events) > max_events:
            events = rng.choice(events, size=max_events, replace=False)

        selected_events.extend(events)

    return df[df[event_col].isin(selected_events)].reset_index(drop=True)



def load_inference_results(filename: str) -> pd.DataFrame:
    data = np.load(filename, allow_pickle=True)

    def make_repr(emb_key, proj_key, files_key):
        d = {
            "emb": data[emb_key].tolist(),
            "rec_path": data[files_key].tolist(),
        }
        if proj_key in data:
            d["proj"] = data[proj_key].tolist()
        return pd.DataFrame(d)

    repr1 = make_repr("emb1", "proj1", "files1")
    repr2 = make_repr("emb2", "proj2", "files2")

    df = pd.concat([repr1, repr2], ignore_index=True)
    labels_name = data.get("dataset")
    if isinstance(labels_name, np.ndarray):
        labels_name = labels_name.item() if labels_name.shape == () else labels_name.tolist()
    return df, labels_name
