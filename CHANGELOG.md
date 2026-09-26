# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-26

### Added

- Optionally label Frigate's unlabeled face snapshots (the Face Library's Train tab) with
  the queued Immich person they clearly match, and export them next to the Immich picks.
  Set `FRIGATE_URL`, plus `FRIGATE_USER` and `FRIGATE_PASSWORD` for an authenticated port.
  A snapshot needs Frigate's recognition score against the person's whole Immich library,
  and a clear lead over every other queued person; one per event is kept. Frigate itself
  isn't modified.

## [0.3.0] - 2026-09-25

A rewrite focused on what actually makes Frigate recognize people: a library that covers
the whole range of someone's photos, cropped exactly as Frigate crops. See
[how it works](docs/how-it-works.md).

### Changed

- **Selection spreads out instead of matching the typical face.** v0.2 chose images whose
  average matched the center of a person's library. That reproduced the library's posed,
  frontal look, which camera faces don't share, and it often stopped after a few images.
  Selection now uses farthest-point sampling from the most typical face, as v0.1's
  diversity selection effectively did, with a safeguard v0.1 lacked. In a simulation with
  Frigate's exact scoring, this recognizes about twice as many camera faces.
- **Faces are cropped and embedded exactly as Frigate stores them**: YuNet's tight box,
  LBF alignment, ArcFace. Previous versions embedded a 15% margin crop, which Frigate never
  sees. Exports are that tight crop, saved as WebP like Frigate's own uploads.
- **Wrong faces are removed first**: faces with cosine below 0.3 to the person's
  geometric median (Frigate's own midpoint), and faces closer to another queued person.
- Sharpness is measured with the face scaled to 112 × 112, as ArcFace sees it. The old
  native-resolution check rejected large, sharp faces and kept small, noisy ones.
- Analysis uses Immich previews. Originals are downloaded only for the exported images,
  and analyses are cached, so reruns are nearly instant.
- A simpler CLI: pick several people at once by number or name, or pass names and
  `--count`, `--years` and `--yes` for scripted runs. The summary shows usable faces, the
  top rejection reasons, and how many held-out photos Frigate would recognize.
- Export folders are named after the person, which is the label Frigate uses.
- Much smaller install: InsightFace, PyTorch, Transformers and SigLIP are gone from face
  mode. The `gpu` extra brings its own CUDA libraries. Object mode is an optional
  `objects` extra that uses YOLO11 for detection and embeddings.

### Fixed

- People beyond the first 500 in Immich were never listed.
- Videos were included in searches.
- Face boxes slightly outside the photo were rejected instead of clipped.
- The saved Immich connection file is now readable only by its owner.

### Removed

- Camera manifests and held-out camera evaluation, time-spread selection, the Starter
  preset, optional 112 × 112 alignment export, and the run log file.

### Upgrading

- Renamed settings: `FACE_MAX_IMAGES` → `MAX_IMAGES`, and `MIN_FACE_WIDTH` → `MIN_FACE_SIZE`.
  The size is now measured in Immich preview pixels, 80 by default. The default
  `BLUR_THRESHOLD` is now 50, measured at 112 × 112.
- Removed settings: `MIN_CONFIDENCE`, `MAX_AUTO_IMAGES`, `FACE_MARGIN`,
  `ENABLE_FACE_ALIGNMENT`, `ENABLE_CACHE`, `FACE_BURST_SECONDS`,
  `FACE_PIXEL_DUPLICATE_DISTANCE`, `FACE_OPTIMIZATION_EPSILON`, `FACE_IDENTITY_MARGIN`,
  `FACE_OUTLIER_MAD`, `FRIGATE_VERSION`, `FRIGATE_MODEL_DIR`, `FRIGATE_UNKNOWN_SCORE`,
  `FRIGATE_BLUR_CONFIDENCE_FILTER`, `CAMERA_MANIFEST`.
- Runs are saved as `frigate_train/<date_time>/<Name>/NNN.webp`. Replace a person's
  earlier images in Frigate rather than adding to them, so the old selection's bias
  doesn't remain.

## [0.2.1] - 2026-09-04

### Fixed

- Evaluate tight face crops with a 320×320 SCRFD input instead of the full-photo
  640×640 scale. Oversized detector inputs caused valid faces to be missed or receive
  artificially low confidence, sometimes leaving only one eligible enrollment image.
  Confidence, target-overlap, resolution, exposure, and blur thresholds are unchanged.
- Distinguish scanned assets, quality-passed faces, eligible candidates, and selected
  images in the preview. A prepared JPEG is no longer presented as an approved face.
- Record stage counts, detector input size, and selection stop reasons in manifests.
  Clarify that reference cosine is not independent recognition confidence, and clamp
  displayed centroid/reference cosine to its valid numerical range.
- Bump the preprocessing version so previous cached processing cannot obscure the fix.

## [0.2.0] - 2026-09-04

### Highlights

This release selects face enrollment images around the **Frigate 0.17.2 large / ArcFace
identity centroid**, with stricter target association, consistent image preparation,
and optional independent camera evaluation. The default is **up to 30 natural face
crops per person**; selection can stop early when additional images do not help.

### Added

- Frigate-compatible landmark alignment, raw ArcFace embeddings, 15% coordinate-wise
  trimmed centroids, confidence conversion, and blur confidence reduction.
- Deterministic subset search using additions, swaps, and removals, with comparisons
  against other queued identities and leave-one-out diagnostics.
- Local camera manifests with separate reference, validation, and test events;
  baseline comparisons, recognition and false-acceptance rates, and reported failures.
- Unique export directories and a versioned JSON manifest with provenance, quality
  measurements, model fingerprints, selection reasons, output paths, and file hashes.
- Automated regression checks and dedicated configuration, selection, and camera
  evaluation guides.

### Changed

- Face presets are now **Centroid**, **Starter**, and **Custom**. Counts are ceilings;
  neither smart selection nor time spread relaxes mandatory quality gates.
- Face selection uses Frigate ArcFace rather than Buffalo_L embeddings. InsightFace
  remains responsible for matching the intended face and local detection confidence.
- Duplicate suppression uses capture evidence instead of an embedding-distance cutoff,
  preserving useful independently captured similar faces. Isolated outliers are filtered.
- Default output uses natural-resolution crops with a 15% margin. Both effective face
  dimensions must reach 100 pixels before margin, alignment, or resizing.
- Refreshed the dependency lockfile and GitHub Actions versions. OpenCV uses the contrib
  headless wheel required for Frigate landmarks; both OpenCV 4 and 5 layouts are supported.
- Reorganized the README around setup, selection, and export, with linked reference guides.

### Fixed

- Wrong-person selection in group photos: metadata and local detections must identify
  one unambiguous target. Missing or ambiguous matches are rejected.
- Quality checks now measure the intended face region. Balanced color images are no
  longer mistaken for grayscale through channel-mean comparisons.
- Prepared crops are encoded once and exported byte-for-byte, avoiding differences
  between evaluated and exported images.
- Unified EXIF orientation and RGB conversion, with preview fallback when originals
  cannot be decoded. Inline people metadata is explicitly requested from Immich.
- Invalid geometry, unresolved edited-image coordinates, and undersized faces cannot
  bypass the gates through resizing or fallback selection.
- Configuration overrides are loaded and validated; embedding caches distinguish
  identities, prepared bytes, model weights, runtime providers, and preprocessing.
- Fresh GPU startup preloads installed CUDA libraries before creating ONNX sessions,
  fixing missing `libcublasLt.so.13` errors without relying on an earlier PyTorch import.
- Exports no longer accumulate stale files in reused person directories. Interrupted
  runs remain visibly incomplete and previous exports are preserved.

### Upgrading

- Run `uv sync --locked`; launch with `uv run --extra gpu if-curator` for NVIDIA GPU use.
- Face mode targets **Frigate 0.17.2 large**. Other recognizer profiles are not verified.
  Match evaluation thresholds and blur settings to your Frigate installation.
- `FACE_DUPLICATE_DISTANCE` has been removed. Capture-based duplicate controls and the
  complete environment reference are documented in `docs/configuration.md`.
- New exports live under `frigate_train/<run-id>/<person>-<identity-hash>/`. Existing
  enrollment folders are untouched; review and enroll the new crops yourself.
- Old face and orientation-sensitive object embedding caches become misses. Optional
  export alignment remains available through `ENABLE_FACE_ALIGNMENT=true`.
- Object selection retains its SigLIP, K-Medoids, farthest-point, and YOLO workflow.

### Validation and scope

- 115 offline regression tests, Ruff, and fresh-process CUDA face inference passed.
  Real CPU and RTX 4090 smoke checks covered InsightFace, Frigate ArcFace, SigLIP, and YOLO;
  a local group-photo export was inspected.
- The subset search is a bounded heuristic, not a global optimum. No improvement in
  recognition accuracy on the user's cameras has been measured. Camera evaluation is
  per crop and excludes Frigate's detector, tracking, and temporal aggregation.

## [0.1.0] - 2026-03-03

### Added
- Initial release of `if-curator` — Immich to Frigate training set curator
- **Face recognition prep**: InsightFace (ArcFace/Buffalo_L) embeddings on face crops for Frigate face recognition training
- **Object/state classification prep**: SigLIP (Vision Transformer) embeddings with YOLOv9c object detection
- **Smart diversity selection**: K-Medoids clustering + Farthest Point Sampling (FPS) with hard-example weighting
- **Adaptive auto-threshold**: Automatically stops selection when adding more images becomes redundant (capped at 80)
- **Quality filtering**: Automatic rejection of blurry, grayscale/IR, over/underexposed, low-confidence, and too-small images
- **Face alignment**: Align faces to ArcFace 112×112 format via InsightFace landmarks
- **Full-resolution downloads**: Downloads originals for final crops (falls back to JPEG preview for HEIC/RAW)
- **Concurrent downloads**: 8 parallel thumbnail workers for performance
- **Embedding cache**: Optional disk-based cache for faster re-runs
- **Multi-person batch mode**: Process multiple people in one session
- **Interactive CLI**: Rich terminal UI with preview summary table before downloading
- **GPU support**: Optional CUDA/ROCm/MPS acceleration via `onnxruntime-gpu` extra
- **Environment variable configuration**: Full control via `.env` or shell environment
