"""Pick people from Immich, choose their images, and export them for Frigate."""

import argparse
import logging
from collections import Counter
from dataclasses import replace
from functools import partial

import requests
from rich.columns import Columns
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from . import __version__
from .config import load_settings, save_connection
from .immich import Immich, ImmichError
from .selection import Job, select

console = Console(highlight=False)

REASONS = {
    "edited": "edited in Immich",
    "several_faces": "tagged twice",
    "no_face_box": "no face box",
    "too_small": "face too small",
    "sample_limit": "over sample limit",
    "download_failed": "download failed",
    "coordinate_mismatch": "box doesn't fit photo",
    "no_face_detected": "Frigate finds no face",
    "blurry": "blurry",
    "too_dark": "too dark",
    "too_bright": "too bright",
    "grayscale": "grayscale",
    "no_landmarks": "no landmarks",
    "unlike_person": "unlike this person",
    "no_object": "object not found",
    "no_match": "matches no one",
    "ambiguous": "could be several people",
    "same_event": "same event",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="if-curator", description=__doc__)
    parser.add_argument("people", nargs="*", help="names from Immich (asks if omitted)")
    parser.add_argument("-n", "--count", type=positive, help="images per person")
    parser.add_argument("--years", type=positive, help="only use photos from the last N years")
    parser.add_argument("--object", metavar="CLASS", help="export crops of this YOLO class instead of faces")
    parser.add_argument("-y", "--yes", action="store_true", help="export without asking")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def find_person(people: list[dict], query: str) -> dict:
    query = query.strip()
    if query.isdigit() and 1 <= int(query) <= len(people):
        return people[int(query) - 1]
    for matches in (
        [p for p in people if p["name"].casefold() == query.casefold()],
        [p for p in people if query.casefold() in p["name"].casefold()],
    ):
        if len(matches) == 1:
            return matches[0]
        if matches:
            names = ", ".join(sorted({p["name"] for p in matches})[:5])
            raise LookupError(f"“{query}” matches several people ({names}). Use a number or the full name.")
    raise LookupError(f"No one in Immich is called “{query}”.")


def choose_people(people: list[dict], names: list[str]) -> list[dict]:
    if names:
        return list({p["id"]: p for p in map(lambda n: find_person(people, n), names)}.values())
    console.print(Columns([f"[dim]{i:>3}[/] {p['name']}" for i, p in enumerate(people, 1)], column_first=True))
    while True:
        answer = Prompt.ask("\nPeople [dim](numbers or names, separated by commas)[/]", console=console)
        try:
            chosen = [find_person(people, part) for part in answer.split(",") if part.strip()]
        except LookupError as error:
            console.print(f"[yellow]{error}")
            continue
        if chosen:
            return list({p["id"]: p for p in chosen}.values())


def summarize(jobs: list[Job], snapshots: bool = False) -> Table:
    faces = any(job.object_class is None for job in jobs)
    columns = ["Photos", "Usable", "Selected"] + ["Recognized"] * faces + ["Frigate"] * snapshots
    table = Table(box=None, header_style="bold", pad_edge=False)
    table.add_column("Person")
    for column in columns:
        table.add_column(column, justify="right")
    table.add_column("Top rejections", style="dim")
    for job in jobs:
        reasons = Counter(REASONS.get(c.reason, c.reason) for c in job.candidates if c.reason)
        row = [job.name, str(len({c.asset_id for c in job.candidates})), str(len(job.eligible)), str(len(job.selected))]
        if faces:
            row.append("–" if job.recognized is None else f"{job.recognized:.0%}")
        if snapshots:
            row.append(str(len(job.snapshots)))
        table.add_row(*row, ", ".join(f"{reason} {count}" for reason, count in reasons.most_common(3)))
    return table


def match_snapshots(jobs: list[Job], model, settings, columns) -> list | None:
    """Label Frigate's snapshots, or explain why not; this step is optional."""
    from .snapshots import Frigate, analyze_snapshots, label_snapshots

    try:
        with Progress(*columns, console=console) as progress:
            task = progress.add_task("Frigate snapshots", total=None)
            frigate = Frigate(settings.FRIGATE_URL, settings.FRIGATE_USER, settings.FRIGATE_PASSWORD)
            snapshots = analyze_snapshots(frigate, model, settings, partial(progress.update, task))
    except requests.RequestException as error:
        console.print(f"[yellow]Skipping Frigate snapshots: couldn't read them from {settings.FRIGATE_URL} ({error}).")
        return None
    label_snapshots(snapshots, jobs, settings.FRIGATE_RECOGNITION_THRESHOLD)
    return snapshots


def connect(settings) -> tuple[Immich, list[dict]]:
    """Ask for whatever connection details are missing, and save them once Immich accepts them."""
    if not settings.IMMICH_URL or not settings.API_KEY:
        console.print("Connect to Immich. This is saved to .immich_config.json.")
    url = settings.IMMICH_URL or Prompt.ask("Immich URL [dim](e.g. http://192.168.1.5:2283)[/]", console=console)
    key = settings.API_KEY or Prompt.ask("API key", password=True, console=console)
    url = url.strip() if "://" in url else f"http://{url.strip()}"
    immich = Immich(url, key.strip())
    with console.status("Connecting to Immich…"):
        people = sorted(immich.people(), key=lambda p: p["name"].casefold())
    if not settings.IMMICH_URL or not settings.API_KEY:
        save_connection(url, key.strip())
    return immich, people


def positive(text: str) -> int:
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError("must be a whole number of at least 1")
    return int(text)


def ask_number(question: str, default: int) -> int:
    while (answer := IntPrompt.ask(question, default=default, console=console)) < 1:
        console.print("[yellow]Enter a number of at least 1.")
    return answer


def load_model(object_class: str | None, settings):
    with console.status("Loading models (the first run downloads about 300 MB)…"):
        if object_class:
            from .objects import ObjectModel, analyze_objects

            model = ObjectModel(settings.CACHE_DIR, settings.FORCE_CPU)
            model.class_id(object_class)
            return model, analyze_objects
        from .faces import analyze_faces
        from .frigate import FrigateFaces

        return FrigateFaces(f"{settings.CACHE_DIR}/frigate", settings.FORCE_CPU), analyze_faces


def run(args: argparse.Namespace) -> int:
    try:
        settings = load_settings()
    except ValueError as error:
        console.print(f"[red]Configuration error: {error}")
        return 2
    immich, people = connect(settings)
    if not people:
        console.print("[red]Immich has no named people yet.")
        return 1

    chosen = choose_people(people, args.people)
    years, count = args.years or settings.YEARS_FILTER, args.count or settings.MAX_IMAGES
    if not args.people:
        years = args.years or ask_number("Years of photos to use", years)
        count = args.count or ask_number("Images per person", count)
    settings = replace(settings, YEARS_FILTER=years)
    jobs = [Job(person, count, args.object) for person in chosen]

    model, analyze = load_model(args.object, settings)
    console.print(f"[dim]Running on {model.device}.[/]")
    columns = TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn()
    with Progress(*columns, console=console) as progress:
        for job in jobs:
            analyze(immich, model, job, settings, partial(progress.update, progress.add_task(job.name, total=None)))
    select(jobs, settings.FRIGATE_RECOGNITION_THRESHOLD)
    snapshots = match_snapshots(jobs, model, settings, columns) if settings.FRIGATE_URL and not args.object else None

    console.print()
    console.print(summarize(jobs, snapshots is not None))
    if any(job.recognized is not None for job in jobs):
        console.print("[dim]Recognized: how many of the other usable photos Frigate would recognize with this set.[/]")
    if snapshots is not None:
        reasons = Counter(REASONS.get(c.reason, c.reason) for c in snapshots if c.reason)
        labeled = sum(len(job.snapshots) for job in jobs)
        details = "".join(f", {reason} {count}" for reason, count in reasons.most_common())
        console.print(f"[dim]Frigate: {labeled} of {len(snapshots)} unlabeled snapshots labeled{details}.[/]")
    total = sum(len(job.selected) + len(job.snapshots) for job in jobs)
    if not total:
        console.print("\n[yellow]Nothing to export.")
        return 1
    question = f"\nExport {total} images to {settings.OUTPUT_DIR}/?"
    if not args.yes and not Confirm.ask(question, default=True, console=console):
        return 0

    from .export import export

    faces = None if args.object else model
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task("Exporting", total=total)
        destination = export(jobs, immich, faces, settings, partial(progress.advance, task))
    console.print(f"Saved to [bold]{destination}[/]")
    if faces:
        console.print(
            "[dim]Copy each person's folder into Frigate's face library (/media/frigate/clips/faces) and restart "
            "Frigate, or upload the images in Frigate's Face Library.[/]"
        )
    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    logging.getLogger("urllib3").setLevel(logging.DEBUG if args.verbose else logging.ERROR)
    try:
        code = run(args)
    except (KeyboardInterrupt, EOFError):
        console.print("\nCancelled.")
        code = 130
    except requests.RequestException as error:
        if args.verbose:
            raise
        detail = error.response.status_code if error.response is not None else type(error).__name__
        console.print(f"[red]Immich request failed ({detail}). Check IMMICH_URL and that Immich is running.")
        code = 1
    except (ImmichError, LookupError, RuntimeError) as error:
        if args.verbose:
            raise
        console.print(f"[red]{error}")
        code = 1
    raise SystemExit(code)
