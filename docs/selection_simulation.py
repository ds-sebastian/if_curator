"""Toy model of how enrollment choice affects Frigate recognition. Needs only numpy.

Each embedding is identity + shared nuisance factors (pose, light, ...) + noise. Library
photos are mostly one posed look from a few sessions; camera faces vary broadly and are
shifted. Each strategy picks 30 library photos per person, Frigate's trimmed mean is
built from them, and we count camera faces recognized at Frigate's 0.9 threshold.

    uv run python docs/selection_simulation.py
"""

import numpy as np

rng = np.random.default_rng(0)
DIM, PEOPLE, FACTORS, COUNT = 512, 6, 24, 30
NUISANCE, CAMERA_SHIFT = 0.35, 1.0
factors = np.linalg.qr(rng.normal(size=(DIM, FACTORS)))[0].T


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def trimmed_mean(x):
    cut = int(0.15 * len(x))
    return np.sort(x, axis=0)[cut : len(x) - cut].mean(axis=0)


def sample(identity, n, camera):
    if camera:
        weights = rng.normal(0, 1, (n, FACTORS))
        weights[:, 4:8] += CAMERA_SHIFT
    else:
        posed = rng.random(n) < 0.7
        weights = np.where(posed[:, None], rng.normal(1.2, 0.35, (n, FACTORS)) * (np.arange(FACTORS) < 4), 0)
        weights += np.where(posed[:, None], 0, rng.normal(0, 1, (n, FACTORS)))
        weights += rng.normal(0, 0.35, (40, FACTORS))[rng.integers(0, 40, n)]  # shooting sessions
    x = identity + NUISANCE * weights @ factors + rng.normal(0, 0.6 / np.sqrt(DIM), (n, DIM))
    return x * rng.uniform(0.8, 1.2, (n, 1))


def geometric_median(u):
    center = u.mean(axis=0)
    for _ in range(100):
        center = np.average(u, axis=0, weights=1 / np.maximum(np.linalg.norm(u - center, axis=1), 1e-9))
    return unit(center)


def farthest_points(u):
    chosen = [int(np.argmax((u @ u.T).sum(axis=1)))]
    closest = u @ u[chosen[0]]
    while len(chosen) < COUNT:
        chosen.append(int(np.argmin(np.where(np.isin(np.arange(len(u)), chosen), np.inf, closest))))
        closest = np.maximum(closest, u @ u[chosen[-1]])
    return chosen


def centroid_match(x, u):
    """v0.2: grow the set while its trimmed mean moves toward the library's center."""
    reference = geometric_median(u)
    chosen = [int(np.argmax(u @ reference))]
    best = unit(trimmed_mean(x[chosen])) @ reference
    while len(chosen) < COUNT:
        score, index = max(
            (unit(trimmed_mean(x[chosen + [j]])) @ reference, j) for j in range(len(x)) if j not in chosen
        )
        if score <= best + 1e-5:
            break
        chosen.append(index)
        best = score
    return chosen


STRATEGIES = {
    "random": lambda x, u: rng.choice(len(x), COUNT, replace=False),
    "v0.2 centroid matching": centroid_match,
    "farthest points": lambda x, u: farthest_points(u),
}


def recognition(contamination, floor):
    results = {name: [] for name in STRATEGIES}
    for _ in range(5):
        identities = unit(unit(rng.normal(size=(PEOPLE, DIM))) + 0.25 * unit(rng.normal(size=DIM)))
        libraries = [sample(identity, 400, camera=False) for identity in identities]
        cameras = [sample(identity, 300, camera=True) for identity in identities]
        for i, x in enumerate(libraries):  # mislabeled look-alikes and garbage detections
            k = int(contamination * len(x))
            x[: k // 2] = sample(identities[(i + 1) % PEOPLE], k // 2, camera=False)
            x[k // 2 : k] = rng.normal(0, 1, (k - k // 2, DIM)) * np.linalg.norm(x[-1]) / np.sqrt(DIM)
        for name, strategy in STRATEGIES.items():
            centers = []
            for x in libraries:
                u = unit(x)
                keep = u @ geometric_median(u) >= floor
                centers.append(trimmed_mean(x[keep][strategy(x[keep], u[keep])]))
            centers = unit(np.array(centers))
            hits = []
            for person, faces in enumerate(cameras):
                scores = unit(faces) @ centers.T
                confident = np.round(1 / (1 + np.exp(-20 * (scores.max(axis=1) - 0.3))), 2) >= 0.9
                hits.append(np.mean((scores.argmax(axis=1) == person) & confident))
            results[name].append(np.mean(hits))
    return {name: float(np.mean(values)) for name, values in results.items()}


if __name__ == "__main__":
    print(f"{'library':<16}{'identity floor':<16}" + "".join(f"{name:>24}" for name in STRATEGIES))
    for contamination in (0.0, 0.1):
        for floor in (-1.0, 0.3):
            row = recognition(contamination, floor)
            label = f"{contamination:.0%} mislabeled"
            print(f"{label:<16}{floor if floor > -1 else 'none'!s:<16}" + "".join(f"{v:>24.0%}" for v in row.values()))
