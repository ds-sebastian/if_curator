"""Interactive CLI for if-curator."""

import logging
import os
from io import BytesIO

import requests
from PIL import Image
from rich import print as rprint
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt

from .config import Config, ConfigManager
from .diversity import select_diverse_assets
from .embeddings import is_embedding_available
from .image_processing import process_face_mode, process_full_mode, process_object_mode
from .immich_api import fetch_all_assets, filter_recent_assets, get_people
from .logging import console, setup_logging
logger = logging.getLogger(__name__)

# Strategy presets: (limit, mode_name)
STRATEGY_PRESETS = {
    "1": ("auto", "Auto Diversity"),
    "2": (30, "Standard (30)"),
    "3": (100, "Broad (100)"),
}


def _get_strategy_choice(has_embedding: bool, entity_type: str) -> tuple[int | str, str]:
    """Prompt user for training strategy and return (limit, selection_mode)."""
    model_name = "InsightFace" if entity_type == "face" else "SigLIP"

    if has_embedding:
        rprint("  [bold]1.[/bold] Auto (Objective Diversity) [green][Recommended][/green]")
        rprint("     [dim]• Dynamically selects images until redundancy starts[/dim]")
        rprint("  [bold]2.[/bold] Standard (30 images)")
        rprint("  [bold]3.[/bold] Broad (100 images)")
        rprint("  [bold]4.[/bold] Custom Count")
        rprint("  [bold]5.[/bold] Skip")

        choice = Prompt.ask("Choice", choices=["1", "2", "3", "4", "5"], default="1")

        if choice == "5":
            return 0, "skip"
        if choice == "4":
            limit = IntPrompt.ask("Enter number of images", default=30)
            mode = "smart" if Confirm.ask("Use Smart Diversity?", default=True) else "time"
            return limit, mode
        if choice in STRATEGY_PRESETS:
            return STRATEGY_PRESETS[choice][0], "smart"
        return 30, "smart"

    # Fallback when embedding model not available
    rprint(f"  [yellow]Note: {model_name} not available. Using Time Spread.[/yellow]")
    rprint("  [bold]1.[/bold] Standard (30 images) [green][Recommended][/green]")
    rprint("  [bold]2.[/bold] Broad (100 images)")
    rprint("  [bold]3.[/bold] Custom Count")
    rprint("  [bold]4.[/bold] Skip")

    choice = Prompt.ask("Choice", choices=["1", "2", "3", "4"], default="1")
    limits = {"1": 30, "2": 100, "3": IntPrompt.ask("Enter number of images", default=30)}
    return limits.get(choice, 0), "time" if choice != "4" else "skip"


def interactive_configure(people: list[dict]) -> list[dict]:
    """Interactive phase: select person, mode, and configure training strategy."""
    valid_people = sorted([p for p in people if p.get("name")], key=lambda x: x["name"])

    if not valid_people:
        rprint("[red]No people found with names in Immich.[/red]")
        return []

    # Select person
    console.print("\n[bold cyan]Select Person to Train:[/bold cyan]")
    for idx, p in enumerate(valid_people, 1):
        console.print(f"  [bold]{idx}.[/bold] {p['name']}")

    p_choice = IntPrompt.ask("Enter Number", choices=[str(i) for i in range(1, len(valid_people) + 1)])
    person = valid_people[p_choice - 1]
    name = person["name"]

    console.print(f"\nSelected: [bold green]{name}[/bold green]")

    # Select training mode
    rprint("\n[bold cyan]Training Mode:[/bold cyan]")
    rprint("  [bold]1.[/bold] Face (Frigate Face Recognition)")
    rprint("  [bold]2.[/bold] Object (Frigate Object Classification)")

    mode_choice = Prompt.ask("Choice", choices=["1", "2"], default="1")
    entity_type = "face" if mode_choice == "1" else "object"

    config = {"name": name, "mode": entity_type}
    if entity_type == "object":
        config["object_class"] = Prompt.ask("Enter Object Class (e.g. dog, cat, car)", default="dog")

    # Fetch and filter assets
    years = IntPrompt.ask("Filter images older than (years)", default=Config.YEARS_FILTER)

    console.print(f"Scanning for {name} ({entity_type})...")
    with console.status("[bold green]Fetching assets...[/bold green]"):
        all_assets = fetch_all_assets(person)
        recent_assets = filter_recent_assets(all_assets, years=years)

    rprint(f"  Found [bold]{len(all_assets)}[/bold] total, [bold]{len(recent_assets)}[/bold] in range ({years} years).")

    if not recent_assets:
        rprint("  [dim]Skipping (0 recent images).[/dim]")
        return []

    # Strategy selection
    has_embedding = is_embedding_available(entity_type)
    rprint(f"\n[bold cyan]Select Training Strategy for {name}:[/bold cyan]")

    limit, selection_mode = _get_strategy_choice(has_embedding, entity_type)
    if selection_mode == "skip":
        return []

    # Perform selection
    selected_assets = _perform_selection(recent_assets, limit, name, selection_mode, entity_type)

    rprint(f"  [green]Queued {len(selected_assets)} images for {name}.[/green]")
    return [{"person": person, "assets": selected_assets, "limit": len(selected_assets), "config": config}]


def _perform_selection(
    assets: list, limit: int | str, name: str, selection_mode: str, entity_type: str
) -> list:
    """Run diversity selection with progress display."""
    if selection_mode == "smart":
        model_display = "InsightFace (face embeddings)" if entity_type == "face" else "SigLIP (visual embeddings)"
        rprint(f"\n[cyan]Using {model_display} for diversity analysis...[/cyan]")

        # Pre-load model to avoid interference with progress bar
        is_embedding_available(entity_type)

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Computing embeddings for {len(assets)} images...", total=None)
            selected = select_diverse_assets(
                assets, limit, name,
                selection_mode=selection_mode,
                entity_type=entity_type,
                progress_callback=lambda c, t: progress.update(task, completed=c, total=t),
            )

        label = f"Auto-diversity selected {len(selected)}" if limit == "auto" else f"Selected {len(selected)}"
        rprint(f"  [green]{label} diverse images.[/green]")
        return selected

    rprint(f"\n[cyan]Using time-spread selection for {limit} images...[/cyan]")
    with console.status(f"[bold]Selecting {limit} images evenly distributed over time...[/bold]"):
        selected = select_diverse_assets(assets, limit, name, selection_mode="time", entity_type=entity_type)
    rprint(f"  [green]Selected {len(selected)} images using time spread.[/green]")
    return selected


def execute_jobs(jobs: list[dict]) -> None:
    """Download and process images for all jobs."""
    if not jobs:
        return

    console.rule("[bold blue]Execution Phase")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        grand_total = sum(j["limit"] for j in jobs)
        overall_task = progress.add_task("[green]Overall Progress", total=grand_total)

        for job in jobs:
            person, assets, config = job["person"], job["assets"], job["config"]
            name, mode = person["name"], config.get("mode", "face")

            job_task = progress.add_task(f"Processing {name}...", total=len(assets))
            person_dir = os.path.join(Config.OUTPUT_DIR, name)
            os.makedirs(person_dir, exist_ok=True)

            count = 0
            for asset in assets:
                try:
                    resp = requests.get(
                        f"{Config.IMMICH_URL}/api/assets/{asset['id']}/thumbnail?size=preview&format=JPEG",
                        headers={"x-api-key": Config.API_KEY, "Accept": "application/json"},
                        timeout=30,
                    )
                    if not resp.ok:
                        progress.console.print(f"[red]Failed download {asset['id']}[/red]")
                    else:
                        img = Image.open(BytesIO(resp.content))
                        saved = (
                            process_face_mode(img, asset, person, person_dir, count, min_width=Config.MIN_FACE_WIDTH)
                            if mode == "face"
                            else process_object_mode(img, config, person_dir, count)
                            if mode == "object"
                            else process_full_mode(img, person_dir, count)
                        )
                        if saved:
                            count += 1
                except Exception as e:
                    logger.error(f"Failed to process asset {asset['id']}: {e}")

                progress.advance(job_task)
                progress.advance(overall_task)

            progress.remove_task(job_task)


def main() -> None:
    """Entry point for if-curator CLI."""
    try:
        setup_logging(verbose=False)

        console.print(r"""
    [bold blue]if-curator[/bold blue]
    [dim]Immich -> Frigate Training Data Curator[/dim]
        """)

        ConfigManager.get().interactive_setup()

        try:
            Config.validate()
        except ValueError as e:
            rprint(f"[bold red]Configuration Error:[/bold red] {e}")
            return

        rprint(f"Server: [dim]{Config.IMMICH_URL}[/dim]")
        rprint(f"Output: [dim]{Config.OUTPUT_DIR}[/dim]")

        people = get_people()
        if not people:
            rprint("[bold red]Could not fetch people from Immich. Check URL/Key.[/bold red]")
            return

        jobs = interactive_configure(people)

        if jobs and Confirm.ask(f"\nReady to process {sum(j['limit'] for j in jobs)} images?"):
            execute_jobs(jobs)
            rprint("\n[bold green]Done! Happy Training.[/bold green]")
        elif not jobs:
            rprint("[yellow]No jobs configured.[/yellow]")

    except KeyboardInterrupt:
        rprint("\n[bold red]Aborted by user.[/bold red]")


if __name__ == "__main__":
    main()
