# Camera reference and evaluation

[← Back to the README](../README.md)

Provide **already cropped, correctly labeled camera faces** in a local JSON manifest:

```bash
uv run if-curator --camera-manifest camera/manifest.json
```

```json
{
  "schema_version": 1,
  "samples": [
    {
      "id": "ref-1",
      "person_id": "IMMICH_PERSON_ID",
      "split": "reference",
      "capture_group": "event-1",
      "path": "ref.jpg"
    },
    {
      "id": "val-1",
      "person_id": "IMMICH_PERSON_ID",
      "split": "validation",
      "capture_group": "event-2",
      "path": "val.jpg"
    },
    {
      "id": "test-1",
      "person_id": "IMMICH_PERSON_ID",
      "split": "test",
      "capture_group": "event-3",
      "path": "test.jpg"
    },
    {
      "id": "unknown-1",
      "person_id": null,
      "split": "test",
      "capture_group": "event-4",
      "path": "unknown.jpg"
    }
  ]
}
```

Paths resolve relative to the manifest. `person_id` is mandatory; `null` explicitly
means someone who is not enrolled. Queue all labeled identities in face mode. Group
all frames/crops from the same camera event under one `capture_group`; groups, file
hashes and optional Immich `asset_id` values must not cross splits. Supply `asset_id`
when a crop comes from an Immich asset so enrollment leakage can be detected. Re-encoded
or recropped versions cannot reliably be recognized as the same event without metadata.

Reference crops determine the desired center. Validation crops tune the subset. Test
crops are embedded **after selection** and never affect it. Camera crops are not exported
as enrollment images in this workflow. Do not use test results to repeatedly tune the
same benchmark; reserve new events for subsequent independent evaluation.

The run manifest reports per-crop correct recognition, wrong-identity acceptance,
unknown false acceptance and inference failures, alongside a one-image baseline.
Failed and difficult crops stay in denominators. An unavailable rate is `null`, never
an invented zero. This does not reproduce Frigate’s detector, tracking or temporal
aggregation, and frame-level rates are not independent event-level statistics.
Match `FRIGATE_UNKNOWN_SCORE`, `FRIGATE_RECOGNITION_THRESHOLD` and the blur policy to
your installation before interpreting results. The supported compatibility profile is Frigate
0.17.2; unverified release numbers fail rather than silently claiming parity. The profile
follows the [stable recognizer](https://github.com/blakeblackshear/frigate/blob/v0.17.2/frigate/data_processing/common/face/model.py)
and [embedding preprocessing](https://github.com/blakeblackshear/frigate/blob/v0.17.2/frigate/embeddings/onnx/face_embedding.py).
