# Week 3 A Features

`probe.data.extract_a_features` builds the observation-and-instruction
baseline for the Week 3 predictor comparison.

## Definition

For each state-condition record, the extractor encodes:

1. `policy_image` from the initial agent-view camera;
2. `policy_wrist_image` from the initial wrist camera; and
3. the record's true `instruction`.

Each projected embedding is L2-normalized independently. The A feature is the
concatenation in this order:

```text
[agent image embedding, wrist image embedding, instruction embedding]
```

The encoder is frozen and must be specified explicitly with `--model-id`.
CLIP or SigLIP models exposing `get_image_features` and
`get_text_features` are supported. Labels, actions, B features, and VLA
policy calls are not read.

## Command

Run this after `build_week3_dataset` has created the combined Week 3 dataset:

```bash
PYTHONPATH="$PROBE_ROOT" \
python -m probe.data.extract_a_features \
  --dataset-dir "$PROBE_ROOT/data/week3_all_suites" \
  --model-id "$PROBE_ROOT/models/clip-vit-base-patch32"
```

The model directory must already contain the processor and model files. The
command is offline by default. To permit a Hugging Face download explicitly,
add `--allow-download`.

If the Week 3 manifest points at a different observation root, add:

```text
--source-root /path/to/observation-root
```

## Outputs

The command writes:

```text
week3_all_suites/
  manifest.jsonl                 # a_feature_path and metadata are added
  a_feature_summary.json
  features/
    A/
      <record_id>.npz
    A_matrix.npz
```

Each feature file contains `a_feature` as a one-dimensional `float32`
vector. The matrix contains one row per manifest record in manifest order.
The summary records the encoder, revision, preprocessing, feature dimension,
and number of unique observations.

The same observation is intentionally encoded once and reused by the three
conditions of a state. The text embedding remains condition-specific.
