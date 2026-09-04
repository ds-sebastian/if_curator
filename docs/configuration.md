# Configuration reference

[← Back to the README](../README.md)

The first run prompts for `IMMICH_URL` and `API_KEY` and stores connection details in
`.immich_config.json`. Environment values override saved connection settings. Processing
settings come from the environment (including `.env`) or these defaults.

### Connection

| Variable | Default | Meaning |
| --- | --- | --- |
| `IMMICH_URL` | **required** | [Immich](https://immich.app) server URL |
| `API_KEY` | **required** | Immich API key |
| `OUTPUT_DIR` | `./frigate_train` | Parent directory for isolated runs |

### Selection and filtering

| Variable | Default | Meaning |
| --- | --- | --- |
| `YEARS_FILTER` | `10` | Default age cutoff in years |
| `MIN_FACE_WIDTH` | `100` | Minimum effective width **and height** before resizing |
| `BLUR_THRESHOLD` | `100.0` | Minimum face-region Laplacian variance |
| `MIN_CONFIDENCE` | `0.7` | Minimum matched local detection confidence |
| `FACE_MAX_IMAGES` | `30` | Default face selection ceiling |
| `FACE_BURST_SECONDS` | `2.0` | Maximum time gap for thumbnail-based burst suppression |
| `FACE_PIXEL_DUPLICATE_DISTANCE` | `0.02` | Mean absolute 16×16 RGB difference, scaled to 0–1 |
| `FACE_OPTIMIZATION_EPSILON` | `0.00001` | Minimum objective improvement for an accepted change |
| `FACE_IDENTITY_MARGIN` | `0.1` | Desired correct-versus-rival cosine margin |
| `FRIGATE_VERSION` | `0.17.2` | Verified large-recognizer profile |
| `FRIGATE_MODEL_DIR` | `.if_cache/frigate` | Checksummed ArcFace and LBF model cache |
| `FRIGATE_UNKNOWN_SCORE` | `0.8` | Per-crop unknown acceptance threshold |
| `FRIGATE_RECOGNITION_THRESHOLD` | `0.9` | Per-crop recognition threshold for evaluation |
| `FRIGATE_BLUR_CONFIDENCE_FILTER` | `true` | Apply Frigate’s whole-crop blur confidence reduction |
| `CAMERA_MANIFEST` | empty | Optional local reference/validation/test manifest; CLI flag takes precedence |
| `FACE_OUTLIER_MAD` | `3.0` | Smart isolation threshold multiplier |
| `REJECT_GRAYSCALE` | `true` | Reject grayscale enrollment crops |
| `MAX_AUTO_IMAGES` | `80` | **Object-mode only** auto-selection ceiling |

### Image handling and runtime

| Variable | Default | Meaning |
| --- | --- | --- |
| `FACE_MARGIN` | `0.15` | Margin on each side as a fraction of face size |
| `USE_FULL_RESOLUTION` | `true` | Prefer originals, with preview fallback |
| `ENABLE_FACE_ALIGNMENT` | `false` | Export locally aligned 112×112 crops instead of natural crops |
| `FORCE_CPU` | `false` | Restrict inference to CPU |
| `ENABLE_CACHE` | `false` | Cache embeddings on disk |
| `CACHE_DIR` | `.if_cache` | Embedding cache directory |

Booleans accept `true/false`, `yes/no`, or `1/0`. Invalid settings fail validation.

Face cache keys include person/face/asset identity, encoded-image hash, model-file fingerprint,
and preprocessing version. Old face cache entries are ignored; caching does not bypass current
quality validation or eliminate image downloads.
