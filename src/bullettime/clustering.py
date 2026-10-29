from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SphericalCluster:
    """Represents a single spherical cluster of 3D points.

    Attributes:
        centroid: The 3D coordinates [x, y, z] representing the cluster center.
        point_indices: Indices of points from the input array assigned to this cluster.
        variance: The mean squared distance from member points to the centroid.
    """

    centroid: np.ndarray
    point_indices: list[int]
    variance: float


@dataclass
class SphericalClusteringResult:
    """The result of spherical 3D point clustering.

    Attributes:
        clusters: List of identified spherical clusters.
        outlier_indices: Indices of points that were not assigned to any cluster.
    """

    clusters: list[SphericalCluster]
    outlier_indices: list[int]

    def __iter__(self):
        """Allows unpacking: clusters, outlier_indices = result."""
        return iter((self.clusters, self.outlier_indices))


def spherical_clustering_3d_points(
    points: np.ndarray | list[list[float]],
    radius: float,
    min_samples: int = 3,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> SphericalClusteringResult:

    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")

    pts = np.asarray(points, dtype=float)
    num_points = len(pts)

    if num_points == 0:
        return SphericalClusteringResult(clusters=[], outlier_indices=[])

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points array of shape (N, 3), got {pts.shape}")

    if num_points < min_samples:
        return SphericalClusteringResult(
            clusters=[],
            outlier_indices=list(range(num_points)),
        )

    # Step 1: Iterative centroid candidate refinement for each point
    candidates: list[dict] = []

    for i in range(num_points):
        current_center = pts[i].copy()
        history: list[tuple[np.ndarray, np.ndarray, float]] = []
        visited_subsets: set[tuple[int, ...]] = set()

        for _ in range(max_iter):
            # Compute Euclidean distance to all points
            dists = np.linalg.norm(pts - current_center, axis=1)
            in_sphere_idx = np.where(dists <= radius)[0]

            if len(in_sphere_idx) == 0:
                break

            subset_key = tuple(sorted(in_sphere_idx.tolist()))
            new_center = np.mean(pts[in_sphere_idx], axis=0)
            variance = float(
                np.mean(np.sum((pts[in_sphere_idx] - new_center) ** 2, axis=1))
            )

            history.append((new_center, in_sphere_idx, variance))

            # Convergence check: centroid movement is negligible
            shift = float(np.linalg.norm(new_center - current_center))
            if shift < tol:
                break

            # Oscillation check: cycle detected in member subsets
            if subset_key in visited_subsets:
                # Select the state in history with minimum variance
                best_state = min(history, key=lambda s: s[2])
                new_center, in_sphere_idx, variance = best_state
                break

            visited_subsets.add(subset_key)
            current_center = new_center
        else:
            # Reached max_iter without convergence: fallback to state with minimum variance
            if history:
                best_state = min(history, key=lambda s: s[2])
                new_center, in_sphere_idx, variance = best_state

        if len(in_sphere_idx) > 0:
            candidates.append(
                {
                    "centroid": new_center,
                    "indices": set(in_sphere_idx.tolist()),
                    "indices_list": sorted(in_sphere_idx.tolist()),
                    "variance": variance,
                }
            )

    # Step 2: Merge identical clusters & filter by min_samples
    unique_candidates = {frozenset(cand["indices"]): cand for cand in candidates}

    # Filter out clusters below min_samples
    valid_candidates = [
        cand
        for cand in unique_candidates.values()
        if len(cand["indices"]) >= min_samples
    ]

    # Step 3: Greedy selection and non-maximum suppression (disjoint clusters)
    # Sort: descending by point count, then ascending by variance (higher density)
    valid_candidates.sort(key=lambda c: (-len(c["indices"]), c["variance"]))

    selected_clusters: list[SphericalCluster] = []

    while valid_candidates:
        best = valid_candidates.pop(0)
        selected_clusters.append(
            SphericalCluster(
                centroid=best["centroid"],
                point_indices=best["indices_list"],
                variance=best["variance"],
            )
        )
        best_set = best["indices"]

        # Discard any candidate cluster sharing 1 or more points with the selected cluster
        valid_candidates = [
            cand for cand in valid_candidates if not (cand["indices"] & best_set)
        ]

    # Step 4: Identify outlier points
    assigned_indices: set[int] = set()
    for cluster in selected_clusters:
        assigned_indices.update(cluster.point_indices)

    all_indices = set(range(num_points))
    outlier_indices = sorted(list(all_indices - assigned_indices))

    return SphericalClusteringResult(
        clusters=selected_clusters,
        outlier_indices=outlier_indices,
    )
