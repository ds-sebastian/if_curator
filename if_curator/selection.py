"""Deterministic bounded subset search against Frigate class centroids.

This is a local search, not an assertion of a globally optimal set or camera accuracy.
Reference, validation and test sets have separate roles; only validation tunes selection.
"""

from datetime import datetime, timezone

import numpy as np

from .config import Config
from .faces import _quality_order
from .frigate import class_mean, confidence, unit

SELECTION_REASONS = {"duplicate_capture", "isolated_outlier", "not_selected_objective", "not_selected_budget"}


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()
    except (ValueError, TypeError):
        return None


def approved_pool(candidates, mode):
    valid = []
    for candidate in candidates:
        candidate.selected, candidate.selection_reason = False, None
        candidate.reasons = [r for r in candidate.reasons if r not in SELECTION_REASONS]
        if candidate.reasons:
            continue
        try:
            unit(candidate.embedding)
        except (ValueError, TypeError):
            candidate.reasons.append("invalid_embedding")
            continue
        valid.append(candidate)
    valid.sort(key=_quality_order)
    if mode == "time":
        return valid
    keep = []
    for candidate in valid:
        duplicate = False
        for other in keep:
            same = (
                candidate.asset_id == other.asset_id
                or (candidate.image_hash is not None and candidate.image_hash == other.image_hash)
                or (candidate.capture_group is not None and candidate.capture_group == other.capture_group)
            )
            a, b = timestamp(candidate.created_at), timestamp(other.created_at)
            burst = a is not None and b is not None and abs(a - b) <= Config.FACE_BURST_SECONDS
            pixels = (
                burst
                and candidate.pixel_signature is not None
                and other.pixel_signature is not None
                and (
                    np.mean(np.abs(candidate.pixel_signature - other.pixel_signature))
                    <= Config.FACE_PIXEL_DUPLICATE_DISTANCE
                )
            )
            if same or (burst and pixels):
                duplicate = True
                break
        if duplicate:
            candidate.reasons.append("duplicate_capture")
        else:
            keep.append(candidate)
    if len(keep) >= 10:
        vectors = np.stack([unit(c.embedding) for c in keep])
        distances = np.clip(1 - vectors @ vectors.T, 0, 2)
        np.fill_diagonal(distances, np.inf)
        isolation = np.partition(distances, 4, axis=1)[:, :5].mean(axis=1)
        median = np.median(isolation)
        mad = np.median(np.abs(isolation - median))
        for c, value in zip(keep, isolation):
            c.measurements["neighbor_distance"] = float(value)
            if mad > 1e-8 and value > median + Config.FACE_OUTLIER_MAD * 1.4826 * mad:
                c.reasons.append("isolated_outlier")
    return [c for c in keep if not c.reasons]


def robust_reference(vectors, groups):
    """Spherical direction of a geometric median, with equal weight per capture group."""
    values = np.stack([unit(v) for v in vectors])
    counts = {g: groups.count(g) for g in set(groups)}
    base = np.array([1 / counts[g] for g in groups])
    center = np.average(values, axis=0, weights=base)
    for _ in range(100):
        weights = base / np.maximum(np.linalg.norm(values - center, axis=1), 1e-6)
        next_center = np.average(values, axis=0, weights=weights)
        if np.linalg.norm(next_center - center) < 1e-7:
            center = next_center
            break
        center = next_center
    try:
        return unit(center)
    except ValueError:
        # A symmetric/antipodal pool has no meaningful center. Use a stable medoid direction.
        return values[np.argmin(np.sum(1 - values @ values.T, axis=1))]


def _boundary(score, reduction=0):
    probability = np.clip(score + reduction, 1e-6, 1 - 1e-6)
    return 0.3 + np.log(probability / (1 - probability)) / 20


def objective(centers, references, validation):
    if not centers:
        return 0.0
    identities = sorted(centers)
    matrix = np.stack([unit(centers[p]) for p in identities])
    total = []
    for p in identities:
        index = identities.index(p)
        scores = matrix @ references[p]
        own = scores[index]
        rival = np.max(np.delete(scores, index)) if len(scores) > 1 else -1
        proxy = 1 - own + max(0, Config.FACE_IDENTITY_MARGIN + rival - own)
        known = [s for s in validation if s.person_id == p and s.embedding is not None]
        if not known:
            total.append(proxy)
            continue
        losses = []
        for sample in known:
            scores = matrix @ unit(sample.embedding)
            own = scores[index]
            rival = np.max(np.delete(scores, index)) if len(scores) > 1 else -1
            losses.append(
                1
                - own
                + max(0, Config.FACE_IDENTITY_MARGIN + rival - own)
                + max(0, _boundary(Config.FRIGATE_RECOGNITION_THRESHOLD, sample.reduction) - own)
            )
        total.append(float(np.mean(losses)) + 0.1 * proxy)
    unknown = [s for s in validation if s.person_id is None and s.embedding is not None]
    if unknown:
        total.append(
            2
            * float(
                np.mean(
                    [
                        max(
                            0, np.max(matrix @ unit(s.embedding)) - _boundary(Config.FRIGATE_UNKNOWN_SCORE, s.reduction)
                        )
                        for s in unknown
                    ]
                )
            )
        )
    return float(np.mean(total))


def evaluate(centers, samples):
    """Per-crop metrics, including failed inference; temporal Frigate tracking is outside scope."""
    rows = []
    ids = sorted(centers)
    matrix = np.stack([unit(centers[p]) for p in ids]) if ids else np.empty((0, 512))
    for sample in samples:
        row = sample.record()
        row.update(
            prediction=None, confidence=None, cosine=None, identity_margin=None, accepted=False, recognized=False
        )
        if sample.embedding is not None and len(ids):
            scores = matrix @ unit(sample.embedding)
            # Frigate compares confidence strictly; sorted identities make ties reproducible here.
            best = int(np.argmax(scores))
            score = confidence(scores[best], sample.reduction)
            row.update(
                prediction=ids[best],
                confidence=score,
                cosine=float(scores[best]),
                accepted=score > Config.FRIGATE_UNKNOWN_SCORE,
                recognized=score >= Config.FRIGATE_RECOGNITION_THRESHOLD,
            )
            if sample.person_id in ids:
                own = ids.index(sample.person_id)
                rivals = np.delete(scores, own)
                row["identity_margin"] = float(scores[own] - np.max(rivals)) if len(rivals) else None
        rows.append(row)
    known = [r for r in rows if r["person_id"] is not None]
    unknown = [r for r in rows if r["person_id"] is None]
    return {
        "scope": "per_crop; excludes detection, tracking and temporal aggregation",
        "known_count": len(known),
        "unknown_count": len(unknown),
        "inference_failures": sum(r["error"] is not None for r in rows),
        "correct_recognition_rate": sum(r["recognized"] and r["prediction"] == r["person_id"] for r in known)
        / len(known)
        if known
        else None,
        "known_false_accept_rate": sum(r["accepted"] and r["prediction"] != r["person_id"] for r in known) / len(known)
        if known
        else None,
        "unknown_false_accept_rate": sum(r["accepted"] for r in unknown) / len(unknown) if unknown else None,
        "samples": rows,
    }


def select_jobs(jobs, samples=()):
    face_jobs = [j for j in jobs if j["config"]["mode"] == "face"]
    pools, references, selections, centers = {}, {}, {}, {}
    if len({j["person"]["id"] for j in face_jobs}) != len(face_jobs):
        raise ValueError("Each identity may be configured only once")
    for job in face_jobs:
        p = job["person"]["id"]
        limit, mode = job["requested_limit"], job["selection_mode"]
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("Face limit must be a positive integer")
        if mode not in {"time", "smart"}:
            raise ValueError("Unknown face selection mode")
        if any(c.person_id != p for c in job["candidates"]):
            raise ValueError("Candidate identity does not match job")
        for c in job["candidates"]:
            for s in samples:
                if (
                    s.sha256 == c.image_hash
                    or s.asset_id == c.asset_id
                    or (c.capture_group is not None and s.capture_group == c.capture_group)
                ):
                    if s.split != "reference":
                        raise ValueError("Enrollment overlaps camera validation/test data")
        pool = approved_pool(job["candidates"], mode)
        pools[p] = pool
        refs = [s for s in samples if s.split == "reference" and s.person_id == p and s.embedding is not None]
        if pool:
            # Equal weighting by capture day reduces vacation/burst prevalence in the proxy reference.
            references[p] = robust_reference(
                [s.embedding for s in refs] if refs else [c.embedding for c in pool],
                [s.capture_group for s in refs] if refs else [c.created_at[:10] or c.asset_id for c in pool],
            )
            if mode == "time":
                ordered = sorted(pool, key=lambda c: (c.created_at, c.asset_id))
                chosen = [ordered[i] for i in np.linspace(0, len(ordered) - 1, min(limit, len(ordered)), dtype=int)]
            else:
                chosen = [min(pool, key=lambda c: (1 - unit(c.embedding) @ references[p], _quality_order(c)))]
            selections[p] = chosen
            centers[p] = class_mean([c.embedding for c in chosen])
        else:
            selections[p] = []
        job["selection_report"] = {
            "algorithm": "frigate_centroid_add_swap_remove_v1" if mode == "smart" else "quality_filtered_time_spread",
            "reference_source": "camera_reference" if refs else "immich_capture_day_balanced_proxy",
            "reference_count": len(refs) if refs else len(pool),
            "quality_approved_pool": len(pool),  # Retained for manifest compatibility; post-selection gates.
            "counts": {
                "scanned": len(job["candidates"]),
                "prepared": sum(c.prepared_path is not None for c in job["candidates"]),
                "quality_passed": sum(
                    c.embedding is not None and all(r in SELECTION_REASONS for r in c.reasons)
                    for c in job["candidates"]
                ),
                "duplicate_captures": sum("duplicate_capture" in c.reasons for c in job["candidates"]),
                "isolated_outliers": sum("isolated_outlier" in c.reasons for c in job["candidates"]),
                "eligible": len(pool),
            },
            "objective_history": [],
            "scope": "local heuristic; no camera accuracy claim without independent test samples",
        }
    validation = [s for s in samples if s.split == "validation"]
    baseline = dict(centers)
    initial_loss = objective(centers, references, validation)
    # Two deterministic coordinate passes let changed rival centroids influence later choices.
    for sweep in range(2):
        for job in sorted(face_jobs, key=lambda j: j["person"]["id"]):
            p = job["person"]["id"]
            pool, chosen = pools[p], selections[p]
            if not chosen or job["selection_mode"] == "time":
                continue
            report = job["selection_report"]
            current = objective(centers, references, validation)
            if not report["objective_history"]:
                report["objective_history"].append(current)
            # All candidates enter addition search. Swaps use a deterministic 128-candidate
            # shortlist (closest to reference, best quality, time spread) to bound runtime.
            nearest = sorted(pool, key=lambda c: (1 - unit(c.embedding) @ references[p], _quality_order(c)))
            chronological = sorted(pool, key=lambda c: (c.created_at, c.asset_id))
            shortlist = {
                c.asset_id: c
                for c in nearest[:64]
                + pool[:32]
                + [chronological[i] for i in np.linspace(0, len(pool) - 1, min(32, len(pool)), dtype=int)]
            }
            report["swap_shortlist_count"] = len(shortlist)
            # Bound exchanges even with tiny epsilon; report whether this cap was reached.
            for iteration in range(job["requested_limit"] + 10):
                best, best_center, best_loss = None, None, current
                chosen_ids = {c.asset_id for c in chosen}

                def consider(proposed):
                    nonlocal best, best_center, best_loss
                    try:
                        center = class_mean([c.embedding for c in proposed])
                        loss = objective({**centers, p: center}, references, validation)
                    except ValueError:
                        return  # An antipodal combination cannot form a usable centroid.
                    if loss < best_loss - Config.FACE_OPTIMIZATION_EPSILON:
                        best, best_center, best_loss = proposed, center, loss

                if len(chosen) < job["requested_limit"]:
                    for candidate in pool:
                        if candidate.asset_id not in chosen_ids:
                            consider(chosen + [candidate])
                for position in range(len(chosen)):
                    if len(chosen) > 1:
                        consider(chosen[:position] + chosen[position + 1 :])
                    for candidate in sorted(shortlist.values(), key=_quality_order):
                        if candidate.asset_id not in chosen_ids:
                            consider(chosen[:position] + [candidate] + chosen[position + 1 :])
                if best is None:
                    break
                chosen, centers[p], current = best, best_center, best_loss
                report["objective_history"].append(current)
            else:
                report["iteration_cap_reached"] = True
            selections[p] = chosen
    for job in face_jobs:
        p = job["person"]["id"]
        selected = sorted(selections[p], key=_quality_order)
        selected_ids = {c.asset_id for c in selected}
        for c in pools[p]:
            c.measurements["reference_cosine"] = float(unit(c.embedding) @ references[p])
            rivals = [float(unit(c.embedding) @ unit(center)) for other, center in centers.items() if other != p]
            if rivals:
                c.measurements["identity_margin"] = float(unit(c.embedding) @ unit(centers[p]) - max(rivals))
            if c.asset_id in selected_ids:
                c.selected = True
                c.selection_reason = job["selection_report"]["algorithm"]
            else:
                c.reasons.append(
                    "not_selected_objective" if job["selection_mode"] == "smart" else "not_selected_budget"
                )
        job["selected_faces"], job["limit"] = selected, len(selected)
        if "assets" in job:
            job["assets"] = [a for a in job["assets"] if a["id"] in selected_ids]
        report = job["selection_report"]
        report["counts"]["selected"] = len(selected)
        report["counts"]["not_selected"] = len(pools[p]) - len(selected)
        report["counts"]["rejected"] = len(job["candidates"]) - len(pools[p])
        if not pools[p]:
            report["stop_reason"] = "no_eligible_faces"
        elif len(selected) >= job["requested_limit"]:
            report["stop_reason"] = "count_ceiling"
        elif len(selected) == len(pools[p]):
            report["stop_reason"] = "eligible_pool_exhausted"
        elif report.get("iteration_cap_reached"):
            report["stop_reason"] = "iteration_cap"
        else:
            report["stop_reason"] = "no_objective_improvement"
        report["initial_joint_objective"] = initial_loss
        report["final_joint_objective"] = objective(centers, references, validation)
        report["centroid_reference_cosine"] = (
            float(np.clip(unit(centers[p]) @ references[p], -1, 1)) if selected else None
        )
        report["leave_one_out"] = []
        for i, c in enumerate(selected):
            row = {"asset_id": c.asset_id, "cosine": None, "confidence": None, "identity_margin": None}
            if len(selected) > 1:
                try:
                    loo = unit(class_mean([v.embedding for k, v in enumerate(selected) if i != k]))
                    similarity = float(unit(c.embedding) @ loo)
                    rivals = [float(unit(c.embedding) @ unit(v)) for key, v in centers.items() if key != p]
                    row.update(
                        cosine=similarity,
                        confidence=confidence(similarity, c.measurements.get("frigate_blur_reduction", 0)),
                        identity_margin=similarity - max(rivals) if rivals else None,
                    )
                except ValueError:
                    row["error"] = "undefined_leave_one_out_centroid"
            report["leave_one_out"].append(row)
    return {
        "centers": centers,
        "baseline_centers": baseline,
        "validation": evaluate(centers, validation),
        "baseline_validation": evaluate(baseline, validation),
    }
