"""Interactive CLI for if-curator."""

import hashlib
import logging

from rich import print as rprint
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .config import Config, ConfigManager
from .diversity import select_diverse_assets
from .embeddings import is_embedding_available
from .faces import FacePipelineError, prepare_face_candidates, select_face_candidates
from .image_processing import process_full_mode, process_object_mode
from .immich_api import fetch_all_assets, fetch_full_image, fetch_preview_image, filter_recent_assets, get_people
from .logging import console, setup_logging
from .runs import RunWorkspace, person_directory

logger = logging.getLogger(__name__)

# Strategy presets: (limit, mode_name)
STRATEGY_PRESETS = {
    "1": ("auto", "Auto Diversity"),
    "2": (30, "Standard (30)"),
    "3": (100, "Broad (100)"),
}


def _get_strategy_choice(has_embedding: bool, entity_type: str) -> tuple[int | str, str]:
    """Prompt user for training strategy and return (limit, selection_mode)."""
    if entity_type == "face":
        if not has_embedding:
            raise FacePipelineError("InsightFace unavailable; cannot validate face candidates")
        rprint(f"  1. Diverse (up to {Config.FACE_MAX_IMAGES}) [Recommended]")
        rprint("  2. Starter (up to 5)")
        rprint("  3. Custom count")
        rprint("  4. Skip")
        choice = Prompt.ask("Choice", choices=["1", "2", "3", "4"], default="1")
        if choice == "4":
            return 0, "skip"
        if choice == "3":
            limit = 0
            while limit <= 0:
                limit = IntPrompt.ask("Maximum images (positive integer)", default=30)
            mode = "smart" if Confirm.ask("Use representative diversity?", default=True) else "time"
            return limit, mode
        return (Config.FACE_MAX_IMAGES if choice == "1" else 5), "smart"

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
    limits = {"1": 30, "2": 100}
    if choice == "3":
        limits["3"] = IntPrompt.ask("Enter number of images", default=30)
    return limits.get(choice, 0), "time" if choice != "4" else "skip"


def _configure_person(person: dict, workspace: RunWorkspace) -> dict | None:
    """Configure training for a single person. Returns job dict or None."""
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
        return None

    # Strategy selection
    has_embedding = is_embedding_available(entity_type)
    rprint(f"\n[bold cyan]Select Training Strategy for {name}:[/bold cyan]")

    limit, selection_mode = _get_strategy_choice(has_embedding, entity_type)
    if selection_mode == "skip":
        return None

    if entity_type == "face":
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), console=console
        ) as progress:
            task = progress.add_task("Preparing and evaluating target faces...", total=None)
            candidates, fingerprint = prepare_face_candidates(
                recent_assets,
                person["id"],
                workspace.preparation_directory(person["id"]),
                lambda c, t: progress.update(task, completed=c, total=t),
            )
            selected = select_face_candidates(candidates, limit, selection_mode)
        selected_ids = {c.asset_id for c in selected}
        selected_assets = [a for a in recent_assets if a["id"] in selected_ids]
        return {
            "person": person,
            "assets": selected_assets,
            "limit": len(selected),
            "config": config,
            "candidates": candidates,
            "selected_faces": selected,
            "model_fingerprint": fingerprint,
            "selection_mode": selection_mode,
            "requested_limit": limit,
            "years_filter": years,
        }
    selected_assets = _perform_selection(recent_assets, limit, name, selection_mode, entity_type)
    return {
        "person": person,
        "assets": selected_assets,
        "limit": len(selected_assets),
        "config": config,
        "selection_mode": selection_mode,
        "requested_limit": limit,
        "years_filter": years,
    }


def interactive_configure(people: list[dict], workspace: RunWorkspace) -> list[dict]:
    """Interactive phase: select person(s), mode, and configure training strategy.

    Supports multi-person batch mode — after configuring one person,
    prompts to add another.
    """
    valid_people = sorted([p for p in people if p.get("name")], key=lambda x: x["name"])

    if not valid_people:
        rprint("[red]No people found with names in Immich.[/red]")
        return []

    jobs = []

    while True:
        # Select person
        console.print("\n[bold cyan]Select Person to Train:[/bold cyan]")
        for idx, p in enumerate(valid_people, 1):
            # Mark already-queued people
            marker = " [dim](queued)[/dim]" if any(j["person"]["id"] == p["id"] for j in jobs) else ""
            console.print(f"  [bold]{idx}.[/bold] {p['name']}{marker}")

        p_choice = IntPrompt.ask("Enter Number", choices=[str(i) for i in range(1, len(valid_people) + 1)])
        person = valid_people[p_choice - 1]

        if any(j["person"]["id"] == person["id"] for j in jobs):
            rprint("[yellow]This person is already queued.[/yellow]")
            continue
        job = _configure_person(person, workspace)
        if job:
            jobs.append(job)
            workspace.record_jobs(jobs)

        # Multi-person: ask to add another
        if not Confirm.ask("\nAdd another person?", default=False):
            break

    return jobs


def _perform_selection(assets: list, limit: int | str, name: str, selection_mode: str, entity_type: str) -> list:
    """Run diversity selection with progress display."""
    if selection_mode == "smart":
        model_display = "InsightFace (face embeddings)" if entity_type == "face" else "SigLIP (visual embeddings)"
        rprint(f"\n[cyan]Using {model_display} for diversity analysis...[/cyan]")

        # Pre-load model to avoid interference with progress bar
        is_embedding_available(entity_type)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Computing embeddings for {len(assets)} images...", total=None)
            selected = select_diverse_assets(
                assets,
                limit,
                name,
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


def _show_preview(jobs: list[dict]) -> None:
    """Show a summary table of all queued jobs before execution."""
    table = Table(title="📋 Training Job Preview", show_header=True, header_style="bold cyan")
    table.add_column("Person", style="bold")
    table.add_column("Mode", style="dim")
    table.add_column("Selected", justify="right")
    table.add_column("Prepared", justify="right")
    table.add_column("Rejected / not selected", justify="right")
    table.add_column("Date Range", style="dim")

    for job in jobs:
        name = job["person"]["name"]
        mode = job["config"].get("mode", "face")
        count = str(job["limit"])

        # Date range
        dates = sorted(a.get("fileCreatedAt", "")[:10] for a in job["assets"] if a.get("fileCreatedAt"))
        date_range = f"{dates[0]} → {dates[-1]}" if len(dates) >= 2 else (dates[0] if dates else "—")

        candidates = job.get("candidates", [])
        prepared = sum(c.prepared_path is not None for c in candidates)
        rejected = sum(bool(c.reasons) for c in candidates)
        table.add_row(
            name,
            mode,
            count,
            str(prepared) if mode == "face" else "—",
            str(rejected) if mode == "face" else "—",
            date_range,
        )

    console.print()
    console.print(table)
    console.print()


def execute_jobs(jobs: list[dict], workspace: RunWorkspace | None = None):
    """Publish an isolated run; faces are copied byte-for-byte from evaluation."""
    if not jobs:
        return None
    workspace = workspace or RunWorkspace(Config.OUTPUT_DIR)
    try:
        for job in jobs:
            mode = job["config"].get("mode", "face")
            if mode == "face":
                workspace.export_faces(job)
            else:
                person = job["person"]
                directory = workspace.path / person_directory(person["name"], person["id"], mode)
                directory.mkdir(exist_ok=False)
                job["object_outputs"] = []
                for count, asset in enumerate(job["assets"]):
                    image = (
                        fetch_full_image(asset["id"])
                        if Config.USE_FULL_RESOLUTION
                        else fetch_preview_image(asset["id"])
                    )
                    if image is None:
                        raise ValueError(f"Could not download asset {asset['id']}")
                    try:
                        if mode == "object":
                            process_object_mode(image, job["config"], str(directory), count)
                        else:
                            process_full_mode(image, str(directory), count)
                    finally:
                        image.close()
                    for output in sorted(directory.glob(f"{count}*.jpg")):
                        # Include only this asset's numeric prefix.
                        if output.stem.split("_")[0] == str(count):
                            job["object_outputs"].append(
                                {
                                    "asset_id": asset["id"],
                                    "output_path": str(output.relative_to(workspace.path)),
                                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                                }
                            )
            workspace.record_jobs(jobs)
        return workspace.publish(jobs)
    except BaseException:
        workspace.record_jobs(jobs)
        workspace.fail()
        raise


def main() -> None:
    """Entry point for if-curator CLI."""
    workspace = None
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

        workspace = RunWorkspace(Config.OUTPUT_DIR)
        jobs = interactive_configure(people, workspace)

        if jobs:
            _show_preview(jobs)
            if Confirm.ask(f"Export {sum(j['limit'] for j in jobs)} selected images?"):
                destination = execute_jobs(jobs, workspace)
                console.print(f"Export complete: {destination}")
            else:
                workspace.fail("cancelled")
        else:
            rprint("[yellow]No jobs configured.[/yellow]")
            workspace.fail("cancelled")

    except KeyboardInterrupt:
        if workspace:
            workspace.fail("interrupted")
        rprint("\n[bold red]Aborted by user.[/bold red]")
    except Exception:
        if workspace:
            workspace.fail()
            console.print(f"Run failed; incomplete artifacts: {workspace.path}")
        raise


if __name__ == "__main__":
    main()
