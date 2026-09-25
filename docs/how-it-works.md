# How if-curator chooses images

[← README](../README.md)

## What Frigate does with your images

Frigate 0.17's face recognition (the default `large` model) is simple:

1. **Enrolling.** Each image in a person's face library is aligned with LBF landmarks
   and embedded with ArcFace. The person's *center* is the coordinate-wise 15% trimmed
   mean of those raw embeddings.
2. **Recognizing.** A camera face (YuNet's detection box, cropped tight) is aligned and
   embedded the same way. Frigate compares it with every person's center by cosine
   similarity and turns the best one into a score: 0.5 at cosine 0.30, and 0.9 (the
   default `recognition_threshold`) at cosine 0.41.

So the only thing your images decide is **where each center points**. Frigate stores an
uploaded image as YuNet's tight crop, so that crop is the image that matters.

## Why the spread matters

A face embedding mixes who someone is with how the photo was taken: pose, lighting,
expression, age, glasses, sharpness. Averaging many images cancels out whatever varies
between them. What they share is left in the center.

A phone library is mostly one kind of photo: posed, frontal, well lit, from a handful of
events. A center built from typical library photos, which v0.2 deliberately chose,
keeps that look, and camera faces don't share it: they're seen from above, at an angle,
small, at dusk. Choosing images that **cover the whole range** of a person's photos
cancels those conditions instead, leaving a center that is closer to identity alone.
That is why v0.1, which picked diverse images, trained Frigate better than v0.2.

`docs/selection_simulation.py` reproduces this with Frigate's exact trimmed mean and
scoring. It uses a toy embedding model: identity plus shared nuisance factors, with
libraries dominated by a posed look and camera faces that vary and are shifted.

| Library | Identity floor | Random | v0.2 centroid matching | Farthest points |
| --- | --- | ---: | ---: | ---: |
| clean | none | 40% | 36% | **73%** |
| clean | 0.3 | 37% | 34% | **69%** |
| 10% mislabeled | none | 37% | 36% | 2% |
| 10% mislabeled | 0.3 | 35% | 34% | **66%** |

*Share of camera faces recognized at score ≥ 0.9, with 30 images per person.*

This is a model, not your cameras. It shows the mechanism, not a guaranteed number. The
last rows show the catch: spreading out picks the most unusual faces first, and in a real
library the most unusual "face" is often someone else. So the floor is required.

## The pipeline

For each person:

1. **Find the face.** Immich provides the person's face box. Photos edited in Immich,
   tagged with the person twice, or with a face smaller than `MIN_FACE_SIZE` preview
   pixels are skipped before anything is downloaded.
2. **Crop it the way Frigate does.** Using Immich's preview, run Frigate's YuNet detector
   on the area around the box. Keep the detection that overlaps Immich's box
   (IoU ≥ 0.5), crop it tight, and skip the photo if there's none. That tight crop is
   what Frigate stores, and embedding a looser crop instead changes the embedding a lot:
   cosine 0.82 for a 15% margin in testing.
3. **Check quality as ArcFace sees it.** Scale the crop to 112 × 112, then reject faces
   that are too dark (mean below 30), too bright (above 225), blurry (Laplacian variance
   below `BLUR_THRESHOLD`) or grayscale. Measuring at a fixed scale matters: at native
   resolution, a large, smooth, sharp face looks blurrier than a small, noisy one.
4. **Embed it** with Frigate's landmark alignment and ArcFace. Results are cached in
   `.if_cache/`, so reruns skip downloads and inference.
5. **Drop faces that aren't this person.** Take a robust center of all the person's faces
   (the geometric median, which a minority of wrong faces can't move) and remove faces
   with cosine below 0.3 to it. That's Frigate's own midpoint, where a match becomes more
   likely than not. When several people are queued, also remove faces that are closer to
   someone else's center, which catches Immich mix-ups between look-alikes.
6. **Spread out.** Start from the most typical face, then repeatedly add the face least
   similar to everything chosen so far (farthest-point sampling). Two faces with cosine
   ≥ 0.9 count as the same look. Selection stops at the requested count, or early when
   every remaining face repeats a look already chosen.
7. **Export.** Download the original (falling back to the preview), re-run YuNet on it,
   and save the tight crop as WebP at quality 100, exactly as Frigate's own upload does.

Libraries over 1,000 usable faces are sampled evenly through time.

## Reading the summary

| Column | Meaning |
| --- | --- |
| Photos | Photos of this person in the chosen years |
| Usable | Faces that passed every check above |
| Selected | Images chosen for export |
| Recognized | Share of the *unselected* usable faces that Frigate would recognize (score ≥ 0.9, and this person rather than another queued one) with the selected set |

"Recognized" is a sanity check on library photos, not a camera accuracy estimate. A low
value means the selected set doesn't represent this person's photos well; look at the
rejection reasons or queue look-alikes together. Every rejection reason is also recorded
in `manifest.json`.
