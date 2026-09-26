# if-curator

Pick the photos from your [Immich](https://immich.app) library that train
[Frigate](https://frigate.video) face recognition best.

- **Diverse, not typical.** Chooses faces that cover every pose, light and age in someone's
  photos, so Frigate's average face for them holds up on cameras. [Why →](docs/how-it-works.md)
- **Cropped the way Frigate crops.** Uses Frigate's own detector, landmarks and ArcFace
  model, and exports the exact face crop Frigate would store.
- **Only the right person.** Skips blurry, dark, grayscale and tiny faces, and faces that
  look like someone else. That includes look-alikes queued in the same run.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and an Immich API key with access to people,
search, faces and assets.

```bash
git clone https://github.com/ds-sebastian/if_curator.git
cd if_curator
uv run if-curator
```

The first run asks for your Immich URL and API key, which are saved to `.immich_config.json`,
and downloads Frigate's face models (about 300 MB). Then pick people by number or name:

```text
  1 Alice      4 Dave
  2 Bob        5 Erin
  3 Carol      6 Frank

People (numbers or names, separated by commas): alice, 3
Years of photos to use (10):
Images per person (30):

Person  Photos  Usable  Selected  Recognized  Top rejections
Alice      812     431        30         96%  face too small 204, blurry 91, Frigate finds no face 40
Carol      377     198        30         93%  face too small 101, blurry 38, grayscale 17

Export 60 images to frigate_train/? [y/n] (y):
```

Or skip the questions: `uv run if-curator Alice Carol --count 30 --years 10 --yes`.

**NVIDIA GPU:** `uv run --extra gpu if-curator`. The CUDA libraries are installed with it.

## Adding the images to Frigate

Each run is saved in its own folder, with a subfolder per person:

```text
frigate_train/2026-09-25_214913/
├── Alice/000.webp …
├── Carol/000.webp …
└── manifest.json
```

Copy each person's folder into Frigate's face library (`/media/frigate/clips/faces/`) and
restart Frigate. The folder name is the name Frigate shows. Copying is the most faithful
option: each file is already the crop Frigate would store. Uploading in Frigate's
**Face Library** also works, but Frigate detects the face again, so its crop will differ
slightly.

Queue people who look alike (siblings, parents and children) in the same run, so faces
Immich mixed up between them are left out. For children, use only the last year or two
of photos: faces from years ago won't help Frigate recognize them today.

## Labeling Frigate's snapshots (optional)

Frigate keeps its recent camera face attempts, mostly unlabeled, in the Face Library's
**Train** tab. These are the most useful training images there are, because they show how
your cameras actually see people. Point if-curator at Frigate and it labels the ones that
clearly match someone you queued:

```dotenv
FRIGATE_URL=http://frigate:5000      # or the authenticated port with the two lines below
FRIGATE_USER=admin
FRIGATE_PASSWORD=…
```

A snapshot gets a person's name only when Frigate would recognize it (score ≥ 0.9)
against that person's whole Immich library, and it matches them clearly better than
anyone else queued. Only the best snapshot from each event is kept, and the rest are
spread out like the Immich picks, up to the same count. They're exported into the
person's folder next to the Immich images, and the summary's **Frigate** column shows
how many there are. Frigate itself isn't changed. Queue everyone who regularly appears
on your cameras, so a snapshot of one person can't be credited to a look-alike.

## Settings

Set these in the environment or in `.env`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `IMMICH_URL`, `API_KEY` | asked on first run | Immich connection |
| `OUTPUT_DIR` | `frigate_train` | Where runs are saved |
| `CACHE_DIR` | `.if_cache` | Models and cached face analyses |
| `YEARS_FILTER` | `10` | Default years of photos to use |
| `MAX_IMAGES` | `30` | Default images per person |
| `MIN_FACE_SIZE` | `80` | Smallest face side, in Immich preview pixels |
| `BLUR_THRESHOLD` | `50` | Minimum sharpness of the face at 112 × 112 |
| `REJECT_GRAYSCALE` | `true` | Skip black-and-white and infrared faces |
| `USE_FULL_RESOLUTION` | `true` | Export from originals rather than previews |
| `FRIGATE_RECOGNITION_THRESHOLD` | `0.9` | Your Frigate `recognition_threshold`, for the Recognized column |
| `FORCE_CPU` | `false` | Don't use the GPU |
| `FRIGATE_URL` | empty | Frigate's address, to label its face snapshots |
| `FRIGATE_USER`, `FRIGATE_PASSWORD` | empty | Frigate login, if its API requires one |

## Object mode

To export crops of an object class instead of faces (for example, a pet in someone's
photos, for Frigate object classification):

```bash
uv run --extra objects if-curator Alice --object dog
```

This uses YOLO11 to find the objects and picks a diverse set of crops the same way.

## Development

```bash
uv sync --locked
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

The tests use fake Immich and model objects, so they need no network or downloads.

[MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
