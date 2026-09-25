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
restart Frigate. The folder name is the name Frigate shows. You can also upload the images
in Frigate's **Face Library**. Either way Frigate ends up with the same crops.

Queue people who look alike (siblings, parents and children) in the same run, so faces
Immich mixed up between them are left out.

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
