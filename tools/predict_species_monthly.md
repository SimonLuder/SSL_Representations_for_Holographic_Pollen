# Monthly species prediction

`tools/predict_species_monthly.py` assigns species to events previously predicted
as known by the seasonal filter. It is standalone and needs NumPy, pandas, and
scikit-learn; it does not import the ActivePollenFilter package or run BYOL again.

## Inputs and modes

- `--train-representations` and `--train-labels`: Poleno embeddings and species labels.
- `--test-representations` and `--test-labels`: Córdoba embeddings and event metadata.
- `--known-predictions-csv`: required preceding filter output. Both the full binary
  predictions CSV and `predictions_known_only.csv` are accepted. Only events with
  `isknown_pred == 1` are selected; events missing from the CSV are excluded.
  When the full CSV contains conflicting predictions for an event, it is excluded.
- `--reference-mode all`: references from every available training species.
- `--reference-mode active`: references from exact species names in the month's
  JSON list, intersected with the available training species.
- `--monthly-species-json`: required in active mode, without a default. The format
  is the existing twelve-month mapping, such as `{"January": ["Species A"], ...}`.
  The file determines which species are eligible; use either the active-known
  export or the all-active export as appropriate. In all mode JSON is not used.

Existing `split` columns are respected. Separate CSVs without that column are
assigned their train/test role in memory. A shared CSV must contain explicit
splits. Test species ground truth is optional; training species is required.
Dates are parsed from `_YYYY-MM-DD_` in event IDs, using the same year/month
selection as the seasonal filter. Only image 0 (`files1` / `emb1`) is used,
matching the filter pipeline. There is no averaging of the two image embeddings.

References are sampled once with at most 80 images per species by default
(`--max-train-per-species`, seed 42), matching the filter's sampling policy.
Prediction uses cosine similarity and an unweighted vote of the 10 closest
references (`--n-neighbors`). References from the same event are excluded.
Fewer eligible references means fewer votes. Vote ties choose the species of
the closest supporting neighbour, unlike the older evaluator's ordering-based
tie handling. Equal similarities use reference order.

## Run using the inputs from slurm_seasonal_inference_76.sh

From the BYOL repository on Linux, with its dependencies available:

```bash
NAS_ROOT=/mnt/nas05/data01/simon_luder
FILTER_REPO="$NAS_ROOT/ActivePollenFilter/Marvel_Unseen_Species_Filter"
REPRESENTATIONS="$NAS_ROOT/BYOL/BYOL_Representations_for_Holographic_Pollen/checkpoints/byol_lit_20260918_142416"
LABELS="$NAS_ROOT/Data_Setup/Pollen_Datasets/data/final"
# Replace JOBID with the completed filtering job's ID.
FILTER_RESULTS="$FILTER_REPO/seasonal_results/slurm_JOBID"

python3 tools/predict_species_monthly.py \
  --year 2024 \
  --train-representations "$REPRESENTATIONS/poleno_inference_results.npz" \
  --train-labels "$LABELS/poleno/combined_test_500.csv" \
  --test-representations "$REPRESENTATIONS/cordoba_inference_results.npz" \
  --test-labels "$LABELS/cordoba/cordoba_filter_events.csv" \
  --known-predictions-csv "$FILTER_RESULTS/predictions_known_only.csv" \
  --reference-mode active \
  --monthly-species-json "$FILTER_REPO/data/poleno_monthly_active_species.json" \
  --output-dir "$FILTER_RESULTS/species_active"
```

For all available reference species, set `--reference-mode all`, omit the JSON
argument, and choose a new output directory, e.g. `species_all`.

To add it after the filter stage in the Slurm job, copy the script to the BYOL
NAS clone first, remove `exec` from the preceding filtering command (otherwise
the shell cannot continue), then use the same container and mounts:

```bash
BYOL_REPO="$NAS_ROOT/BYOL/BYOL_Representations_for_Holographic_Pollen"
SPECIES_REFERENCE_MODE=active
SPECIES_MONTHLY_JSON="$MONTHLY_SPECIES_JSON"

singularity exec --cleanenv --pwd /app \
  --bind "$NAS_ROOT:$NAS_ROOT:ro" \
  --bind "$REPO_PATH:/app:rw" \
  "$SIF_PATH" \
  python3 "$BYOL_REPO/tools/predict_species_monthly.py" \
  --year "$YEAR" \
  --train-representations "$REPRESENTATIONS/poleno_inference_results.npz" \
  --train-labels "$LABELS/poleno/combined_test_500.csv" \
  --test-representations "$REPRESENTATIONS/cordoba_inference_results.npz" \
  --test-labels "$LABELS/cordoba/cordoba_filter_events.csv" \
  --known-predictions-csv "$OUTPUT_DIR/predictions_known_only.csv" \
  --reference-mode "$SPECIES_REFERENCE_MODE" \
  --monthly-species-json "$SPECIES_MONTHLY_JSON" \
  --output-dir "$OUTPUT_DIR/species_$SPECIES_REFERENCE_MODE"
```

## Outputs and limits

The output directory must be new. It contains twelve monthly CSVs,
`species_predictions_all_months.csv`, and `summary.json` with reference counts,
configured/missing species, and per-month status. CSVs include event ID, image
path, capture date, `predicted_species`, `neighbors_used`, `vote_fraction`,
`nearest_similarity`, and `prediction_status`. `vote_fraction` is the fraction
of neighbours voting for the winner, not a calibrated probability.

If there are no references for a month, or all references belong to the query
event, matching test rows are exported with blank species predictions and an
explicit status. Months without selected test labels produce header-only CSVs.
Accuracy is calculated only for successfully predicted rows with species ground
truth; otherwise it is null. Events without matching embeddings cannot be
predicted. NPZ parts should be disjoint.

Prediction is batched (`--prediction-batch-size`, default 1000), and only selected
image-0 embedding rows are read. Larger batches use more memory. Slurm execution
and the container environment must be validated on the cluster.
