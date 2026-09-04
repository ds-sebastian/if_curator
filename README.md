# if-curator

Prepare representative face enrollment images from an Immich library for Frigate.
The default exports **up to 30 natural face crops per person**, with quality measured
on the intended face. Object preparation using SigLIP and YOLO is also available.

## What this tool does

Immich provides person labels and face bounding boxes. if-curator uses a local
InsightFace Buffalo_L detector and recognition model to validate each target crop,
remove near-duplicates and isolated outliers, and select representative medoids.
It prepares each JPEG once, evaluates that exact image, and exports its bytes
unchanged into a new run directory.

This prepares enrollment examples; it does not train or improve Frigate's upstream
face detector or fine-tune ArcFace weights. Buffalo_L is a **curation proxy**, not
Frigate's exact recognition pipeline. Its cosine distances are not Frigate
confidence scores, and this tool does not certify recognition accuracy.

## Installation

Requires Python 3.12+, [uv](https://astral.sh/uv/), and an Immich API key able to read
people, search assets, retrieve face metadata, and download originals/previews.

```bash
uv sync
uv run if-curator
```

For GPU inference:

```bash
uv sync --extra gpu
```

ONNX Runtime selects available execution providers; CPU fallback is supported.
The first face run needs the Buffalo_L model pack, downloaded by InsightFace if
not already cached. Set `FORCE_CPU=true` to restrict inference to CPU.

## Face workflow

1. Select an Immich person, face mode, and date range.
2. Choose **Diverse (up to 30)**, **Starter (up to 5)**, or a custom positive count.
   Custom counts can use representative diversity or explicit time spread.
3. The tool prepares and evaluates candidates before showing the selection summary.
4. Review selected/prepared/rejected counts, then export the prepared images.
5. Add another person within the same run when needed.

Counts are ceilings. A shortage never relaxes quality gates or fills from rejected
images. Local face detection remains mandatory in time-spread mode. Model or
selection failure never silently falls back to an unfiltered face export.

### Preparation and quality

- Metadata must identify exactly one face for the requested person. If nested
  asset metadata is incomplete, the tool retrieves `/api/faces?id=…`. Asset searches
  explicitly request people metadata to avoid unnecessary per-asset requests.
- A local detection must match that target box with intersection-over-union of at
  least 0.5. Missing or ambiguous matches are rejected, including montages with
  multiple faces assigned to the same person.
- Originals are fully decoded, EXIF-oriented, and converted to RGB. Unsupported
  formats and failed original downloads fall back to a JPEG preview.
- Coordinates are scaled to the decoded representation. Invalid, incomplete, or
  incompatible boxes are rejected. Assets marked as edited are conservatively
  rejected because their box/original coordinate relationship is not yet verified.
- Both effective face dimensions must be at least 100 pixels **before** margin,
  resizing, or alignment. Preview fallback must independently satisfy this rule.
- Natural-resolution crops have a 15% margin by default. Optional 112×112 alignment
  uses local landmarks, followed by validation of the resulting encoded crop.
  Target landmarks are projected through that transform; the tight aligned crop
  does not undergo a second detector pass. Its detection confidence comes from
  the matched face before alignment.
- Blur, exposure, and grayscale checks operate on the face region of the encoded
  JPEG, excluding the margin/background. Grayscale detection compares channels at
  each pixel, rather than comparing global channel averages.
- Detection confidence comes from the matched local detector. Public Immich face
  metadata is not assumed to provide confidence, landmarks, or embeddings.

The manifest records measured sharpness, brightness, channel differences, and
local detection confidence. These are quality heuristics; the tool does not claim
to measure occlusion or landmark confidence.

### Representative selection

Smart face selection uses the following sequence:

1. Apply all geometry and quality gates and validate the 512-dimensional embedding.
2. Remove near-duplicates whose normalized cosine distance is below `0.05`, retaining
   the candidate with better detection confidence, then sharpness, then resolution.
3. With at least ten remaining candidates, measure mean distance to each face's five
   nearest neighbors. Reject isolated candidates above the median plus three scaled
   median absolute deviations (MAD scale factor 1.4826). Skip this gate when the
   dispersion is zero.
4. Choose up to the requested number with deterministic K-Medoids, reducing total
   cosine distance to representatives. Stable IDs break remaining quality ties.

There is no farthest-point filling, low-confidence boost, or fixed 80/20 allocation
for face mode. Supported appearance variations can remain represented without
rewarding rarity alone. Thresholds are configurable heuristics, not calibrated
Frigate recognition thresholds. Incorrect Immich labels and coherent groups of
mislabeled faces may still pass; visually review enrollment images.

Explicit time spread selects only from quality-approved, successfully embedded
faces; it does not apply smart duplicate/isolation filtering.

### Output and reruns

Each run receives a unique timestamp/ID directory:

```text
frigate_train/
  20260904T123000000000Z-abcd1234/
    manifest.json
    Person-<identity-hash>/
      000.jpg
      001.jpg
```

Folder suffixes distinguish people with identical names. Consult the manifest for
original person names and IDs when enrolling images in Frigate.

During preparation, a hidden `.<run-id>.incomplete` directory holds staged crops
and the manifest. A completed run is published by renaming its directory. Failed,
interrupted, or cancelled runs remain marked incomplete; they are not enrollment
sets. Prior runs and existing manually curated images are never overwritten or
cleaned automatically. Temporary prepared crops are removed when publishing;
rejection reasons remain in the manifest.

The versioned manifest includes processing settings (excluding credentials), model
fingerprint, candidate provenance, source/effective dimensions, quality scores,
selection and rejection reasons, output paths, and SHA-256 hashes. Up to 3,000 assets
per person are sampled evenly through time; excluded assets are recorded. Downloads
use windows of eight workers and stage decoded sources on disk, keeping full-size
images out of the in-memory candidate collection. Preparation can require substantial
network traffic; `USE_FULL_RESOLUTION=false` uses previews with the same gates.

## Configuration

The first run prompts for `IMMICH_URL` and `API_KEY` and stores connection details
in `.immich_config.json`. Environment values override saved connection settings.
Processing settings come from the environment (including `.env`) or these defaults.

| Variable | Default | Meaning |
| --- | --- | --- |
| `IMMICH_URL` | required | Immich server URL |
| `API_KEY` | required | Immich API key |
| `OUTPUT_DIR` | `./frigate_train` | Parent directory for isolated runs |
| `YEARS_FILTER` | `10` | Default age cutoff in years |
| `FORCE_CPU` | `false` | Restrict inference to CPU |
| `MIN_FACE_WIDTH` | `100` | Minimum effective width **and height** before resizing |
| `BLUR_THRESHOLD` | `100.0` | Minimum face-region Laplacian variance |
| `MIN_CONFIDENCE` | `0.7` | Minimum matched local detection confidence |
| `FACE_MAX_IMAGES` | `30` | Default face selection ceiling |
| `FACE_DUPLICATE_DISTANCE` | `0.05` | Smart cosine-distance duplicate threshold; 0 disables removal |
| `FACE_OUTLIER_MAD` | `3.0` | Smart isolation threshold multiplier |
| `REJECT_GRAYSCALE` | `true` | Reject grayscale enrollment crops |
| `FACE_MARGIN` | `0.15` | Margin on each side as a fraction of face size |
| `USE_FULL_RESOLUTION` | `true` | Prefer originals, with preview fallback |
| `ENABLE_FACE_ALIGNMENT` | `false` | Export locally aligned 112×112 crops instead of natural crops |
| `ENABLE_CACHE` | `false` | Cache embeddings on disk |
| `CACHE_DIR` | `.if_cache` | Embedding cache directory |
| `MAX_AUTO_IMAGES` | `80` | **Object-mode only** auto-selection ceiling |

Booleans accept `true/false`, `yes/no`, or `1/0`. Invalid settings fail validation.
Face cache keys include person/face/asset identity, encoded-image hash, model-file
fingerprint, and preprocessing version. Old face cache entries are ignored; caching
does not bypass current quality validation or eliminate image downloads.

## Using the results with Frigate

[Frigate's guidance](https://docs.frigate.video/configuration/face_recognition/)
recommends starting with 1–5 clear frontal images, then expanding methodically with
correctly labeled camera examples. The Starter preset limits count; it does not
certify frontal pose, so review the selected crops. The default diverse set is a
pool of up to 30 enrollment candidates, not a recommendation to import all at once.

Frigate's large ArcFace pipeline uses its own model and alignment/scoring; its small
mode uses FaceNet. Immich's person clustering has a different purpose again.
Recognition performance on camera imagery requires evaluation in that environment.
This iteration does not connect to Frigate, compare household identities, import
camera attempts, or run a Frigate holdout benchmark.

## Object mode

Object mode retains SigLIP embeddings, K-Medoids plus farthest-point selection,
and YOLO object cropping. Its existing Auto, Standard, Broad, and Custom strategies
are separate from face policies. Exports use the same isolated run directories.
All face/object loaders share EXIF correction and RGB decoding. Older SigLIP cache
entries are ignored because their embeddings may reflect uncorrected orientation.
This prepares images for external workflows; it does not upload or train a custom
Frigate object detector.

## Development

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
```

Tests mock APIs and embeddings so they run without network access or model downloads.
