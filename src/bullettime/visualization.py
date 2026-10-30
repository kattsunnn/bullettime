import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _set_axes_equal_3d(ax: plt.Axes) -> None:
    """Set 3D plot axes to equal scale so that shapes are not distorted."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    max_span = np.max(spans)
    centers = np.mean(limits, axis=1)

    half_span = max_span / 2.0 if max_span > 0 else 1.0
    ax.set_xlim3d([centers[0] - half_span, centers[0] + half_span])
    ax.set_ylim3d([centers[1] - half_span, centers[1] + half_span])
    ax.set_zlim3d([centers[2] - half_span, centers[2] + half_span])


def visualize_camera_pose_and_3d_points(
    extrinsics_data: np.ndarray,
    points_3d: np.ndarray,
    axis_scale: float = 0.5,
    title: str = "Camera Poses and 3D Points",
    show_labels: bool = True,
    show_point_indices: bool = True,
    point_size: float = 20.0,
    point_color: str = "black",
) -> None:
    """
    Visualize camera poses and 3D points in a 3D coordinate space.

    Args:
        extrinsics_data: Camera extrinsics as an np.ndarray of shape (N, 4, 3) or (4, 3),
                         where [:3, :] is rotation matrix R and [3, :] is translation vector t.
                         Other shapes (such as (3, 4)) or non-numpy types are rejected.
        points_3d: 3D points as an np.ndarray of shape (N, 3) or (3,). Non-numpy types or
                   invalid shapes are rejected.
        axis_scale: Length of the camera coordinate axes.
        title: Title of the 3D plot.
        show_labels: Whether to show camera index labels at camera centers.
        show_point_indices: Whether to show index numbers next to each 3D point.
        point_size: Marker size for 3D points.
        point_color: Color for 3D points.
    """
    # 1. Reject non-numpy types
    if not isinstance(extrinsics_data, np.ndarray):
        raise TypeError(
            f"extrinsics_data must be a numpy.ndarray, got {type(extrinsics_data).__name__}."
        )
    if not isinstance(points_3d, np.ndarray):
        raise TypeError(
            f"points_3d must be a numpy.ndarray, got {type(points_3d).__name__}."
        )

    # 2. Strict shape validation for extrinsics_data: accept ONLY (4, 3) or (N, 4, 3)
    if extrinsics_data.ndim == 2:
        if extrinsics_data.shape != (4, 3):
            raise ValueError(
                f"Single camera extrinsic must have shape (4, 3), got {extrinsics_data.shape}. Shape (3, 4) is rejected."
            )
        extrinsics = extrinsics_data[np.newaxis, :, :].astype(float)
    elif extrinsics_data.ndim == 3:
        if extrinsics_data.shape[1:] != (4, 3):
            raise ValueError(
                f"Multiple camera extrinsics must have shape (N, 4, 3), got {extrinsics_data.shape}. Shape (3, 4) is rejected."
            )
        extrinsics = extrinsics_data.astype(float)
    else:
        raise ValueError(
            f"extrinsics_data must have 2 or 3 dimensions with shape (4, 3) or (N, 4, 3), got {extrinsics_data.shape}."
        )

    # 3. Strict shape validation for points_3d: accept ONLY (N, 3) or (3,)
    if points_3d.ndim == 1:
        if points_3d.shape != (3,):
            raise ValueError(
                f"Single 3D point must have shape (3,), got {points_3d.shape}."
            )
        pts = points_3d.reshape(1, 3).astype(float)
    elif points_3d.ndim == 2:
        if points_3d.shape[1] != 3:
            raise ValueError(
                f"points_3d must have shape (N, 3), got {points_3d.shape}."
            )
        pts = points_3d.astype(float)
    else:
        raise ValueError(
            f"points_3d must have shape (N, 3) or (3,), got shape {points_3d.shape}."
        )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot 3D points
    if len(pts) > 0:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=point_color,
            s=point_size,
            marker="o",
            alpha=0.8,
            label="3D Points",
        )
        if show_point_indices:
            for idx, pt in enumerate(pts):
                ax.text(
                    pt[0],
                    pt[1],
                    pt[2],
                    f" {idx}",
                    fontsize=8,
                    color=point_color,
                )

    # Plot Camera poses and coordinate axes
    camera_centers = []
    for i, ext in enumerate(extrinsics):
        # Format is strictly (4, 3): R is (3, 3) and t is (3,)
        R = ext[:3, :]
        t = ext[3, :]

        # Camera position in world coordinates: C = -R^T @ t
        camera_center = -R.T @ t
        camera_centers.append(camera_center)

        # Camera coordinate axes directions in world coordinates:
        # X_axis = R^T @ [1, 0, 0]^T = R^T[:, 0]
        # Y_axis = R^T @ [0, 1, 0]^T = R^T[:, 1]
        # Z_axis = R^T @ [0, 0, 1]^T = R^T[:, 2]
        x_axis = R.T[:, 0]
        y_axis = R.T[:, 1]
        z_axis = R.T[:, 2]

        # Draw coordinate axes using quiver (X: Red, Y: Green, Z: Blue)
        ax.quiver(
            camera_center[0],
            camera_center[1],
            camera_center[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            length=axis_scale,
            color="red",
            normalize=True,
            arrow_length_ratio=0.2,
            label="Camera X-axis" if i == 0 else "",
        )
        ax.quiver(
            camera_center[0],
            camera_center[1],
            camera_center[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            length=axis_scale,
            color="green",
            normalize=True,
            arrow_length_ratio=0.2,
            label="Camera Y-axis" if i == 0 else "",
        )
        ax.quiver(
            camera_center[0],
            camera_center[1],
            camera_center[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            length=axis_scale,
            color="blue",
            normalize=True,
            arrow_length_ratio=0.2,
            label="Camera Z-axis" if i == 0 else "",
        )

        if show_labels:
            ax.text(
                camera_center[0],
                camera_center[1],
                camera_center[2],
                f" Cam {i}",
                fontsize=9,
            )

    # Plot camera origins
    if len(camera_centers) > 0:
        c_centers = np.array(camera_centers)
        ax.scatter(
            c_centers[:, 0],
            c_centers[:, 1],
            c_centers[:, 2],
            c="black",
            s=40,
            marker="^",
            label="Camera Origin",
        )

    ax.set_xlabel("X (World)")
    ax.set_ylabel("Y (World)")
    ax.set_zlabel("Z (World)")
    ax.set_title(title)

    # Set equal aspect ratio so 3D perspective isn't distorted
    _set_axes_equal_3d(ax)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# Alias
visualize_cameras_and_3d_points = visualize_camera_pose_and_3d_points
