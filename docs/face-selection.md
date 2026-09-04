# Face preparation and selection

[← Back to the README](../README.md)

The supported compatibility profile is **Frigate 0.17.2 large**. InsightFace verifies
the intended face; Frigate ArcFace provides the embeddings used for selection.

## Preparation and quality

| Stage | Rule |
| --- | --- |
| **Identity** | Metadata must identify exactly one face for the requested person. If nested asset metadata is incomplete, the tool retrieves `/api/faces?id=…`. Asset searches explicitly request people metadata to avoid unnecessary per-asset requests. |
| **Box match** | A local detection must match the target box with intersection-over-union of at least `0.5`. Missing or ambiguous matches are rejected, including montages with multiple faces assigned to the same person. |
| **Decoding** | Originals are fully decoded, EXIF-oriented, and converted to RGB. Unsupported formats and failed original downloads fall back to a JPEG preview. |
| **Coordinates** | Coordinates are scaled to the decoded representation. Invalid, incomplete, or incompatible boxes are rejected. Assets marked as edited are conservatively rejected because their box/original coordinate relationship is not yet verified. |
| **Size** | Both effective face dimensions must be at least 100 pixels **before** margin, resizing, or alignment. Preview fallback must independently satisfy this rule. |
| **Cropping** | Natural-resolution crops have a 15% margin by default. Optional 112×112 alignment uses local landmarks, followed by validation of the resulting encoded crop. Target landmarks are projected through that transform; the tight aligned crop does not undergo a second detector pass — its detection confidence comes from the matched face before alignment. |
| **Quality** | Blur, exposure, and grayscale checks operate on the face region of the encoded JPEG, excluding the margin/background. Grayscale detection compares channels at each pixel, rather than comparing global channel averages. |
| **Confidence** | Detection confidence comes from the matched local detector. Public Immich face metadata is not assumed to provide confidence, landmarks, or embeddings. |

The manifest records measured sharpness, brightness, channel differences, and local
detection confidence. These are quality heuristics; the tool does not claim to measure
occlusion or landmark confidence.

## Centroid selection

Smart face selection uses this deterministic pipeline:

1. **Gate** geometry, size, exposure, blur and local detection confidence; embed the
   exact prepared JPEG using Frigate’s large recognizer. Raw 512-dimensional outputs
   are retained because normalization before coordinate trimming changes the centroid.
2. **Deduplicate captures**, using identical prepared bytes, asset/stack/burst identity,
   or near-identical 16×16 color thumbnails within two seconds. Keep the better crop
   by detection confidence, facial sharpness, effective resolution, then stable IDs.
   Independently captured similar faces remain eligible. No embedding-distance dedupe.
3. **Filter isolation** using five-neighbor mean cosine distance and median plus three
   scaled MADs when at least ten candidates remain; skip when dispersion is zero.
4. **Build a reference** from correctly labeled camera reference crops when supplied.
   Otherwise use a geometric median of normalized Immich embeddings with equal total
   weight per capture day. This reduces the influence of repeated shooting sessions.
5. **Search subsets**, starting at the reference-nearest face and rebuilding Frigate’s
   15% coordinate-wise trimmed mean for additions, swaps and removals. With validation
   crops, minimize correct-identity cosine loss, competing-identity margin violations,
   recognition-threshold shortfalls and unknown false-accept risk. Without them,
   minimize reference-direction loss and ambiguity against other queued identities.
6. **Stop on no improvement**, up to the requested ceiling. Additions search the full
   surviving pool; swaps search at most 128 candidates drawn from reference proximity,
   quality and time spread. Two passes across identities and bounded exchanges keep
   runtime finite. The manifest reports the shortlist and any iteration cap reached.

This is a bounded local search, not a globally optimal solution. Similarity to your own
full enrollment centroid is not independent validation. The manifest instead includes
**leave-one-out** similarity, Frigate confidence and rival margin for each selected
image; a singleton cannot supply a leave-one-out score. Candidate reference similarity,
isolation and rival margins are also recorded. No fixed diversity ratio is imposed.

The objective averages identity-level losses. Its cosine margin defaults to `0.1`;
validation adds a `0.1` reference regularizer and gives unknown threshold violations
weight `2`. These are curation heuristics, not statistically calibrated constants.
Incorrect labels and coherent mislabeled clusters can still survive: review the crops.
Time spread uses quality-approved embeddings and the requested ceiling, without smart
capture deduplication, isolation filtering or centroid optimization.

See [configuration](configuration.md) for thresholds and
[camera evaluation](camera-evaluation.md) for independent reference and test data.

## Library size and storage

Each person’s candidate pool is capped at 3,000 assets sampled evenly through time;
excluded assets are recorded. Downloads run in windows of eight workers, with decoded
sources staged on disk to keep full-resolution images out of the candidate collection.
Use `USE_FULL_RESOLUTION=false` to work from previews, which must pass the same gates.

Temporary prepared crops are removed when a completed run is published. The versioned
manifest retains candidate provenance, source/effective dimensions, measurements,
selection and rejection reasons, model fingerprints, output paths and SHA-256 hashes.
Processing settings are recorded without connection credentials.
