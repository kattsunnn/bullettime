import numpy as np
from omni_directional_img_utils.e2p import E2P
from .types import CropRecord

def calculate_gaze_rays(crop_records: list[CropRecord], extrinsics_data: np.ndarray) -> list[dict]:
    """各Cropレコードに対して、世界座標系におけるカメラ原点と単位方向ベクトルを計算する"""
    gaze_rays = []
    for i, record in enumerate(crop_records):
        cam_idx = record.camera_id
        if cam_idx >= len(extrinsics_data):
            print(f"Warning: camera_id {cam_idx} is out of bounds for extrinsics_data.")
            continue
        # 1. E2Pからunit_sphereを計算
        deg = record.gaze_point_omni_deg
        x, y, z = E2P.angle_to_unit_sphere(deg[0], deg[1])
        X_c = np.array([x, y, z])
        # 2. 外部パラメータの取得
        extrinsic = extrinsics_data[cam_idx]
        R = extrinsic[:3, :]
        t = extrinsic[3, :]
        # 3. 外部パラメータからcamera_centerを計算
        camera_center = -R.T @ t
        # 4. Xw＝R^T Xc -R^T t の関係を利用し、unit_sphereの世界座標を計算
        X_w = R.T @ X_c - R.T @ t
        # 5. 計算した座標 - カメラ原点座標を計算し単位方向ベクトルを計算
        direction = X_w - camera_center
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        # 6. カメラ座標原点と単位方向ベクトルをセットにした直線のパラメータを作成
        ray = {
            "camera_id": cam_idx,
            "origin": camera_center,
            "direction": direction
        }
        gaze_rays.append(ray)
        
        print(f"Record {i} (Cam {cam_idx}): origin = {camera_center}, direction = {direction}")
        
    return gaze_rays


def calculate_pseudo_intersections(gaze_rays: list[dict], d_max: float, return_pairs: bool = False) -> list[np.ndarray] | tuple[list[np.ndarray], list[tuple[int, int]]]:
    intersections = []
    contrib_pairs = []
    n = len(gaze_rays)
    if n < 2:
        if return_pairs:
            return intersections, contrib_pairs
        return intersections

    # 全てのペア (i, j) について距離を計算し、d_max 以下なら疑似交点を算出
    for i in range(n):
        for j in range(i + 1, n):
            if gaze_rays[i]["camera_id"] == gaze_rays[j]["camera_id"]:
                # 同じカメラのペアはスキップ
                continue
            p_i = gaze_rays[i]["origin"]
            v_i = gaze_rays[i]["direction"]
            p_j = gaze_rays[j]["origin"]
            v_j = gaze_rays[j]["direction"]
            # 法線ベクトル n_ij = v_i x v_j
            n_ij = np.cross(v_i, v_j)
            n_norm = np.linalg.norm(n_ij)
            if n_norm < 1e-8:
                # 直線同士が平行な場合はスキップ
                continue
            # 最小距離 d_ij
            d_ij = np.abs(np.dot(p_i - p_j, n_ij)) / n_norm
            
            if d_ij > d_max:
                continue

            v_i_dot_v_j = np.dot(v_i, v_j)
            v_i_dot_v_i = np.dot(v_i, v_i)
            v_j_dot_v_j = np.dot(v_j, v_j)
            
            denom = v_i_dot_v_i * v_j_dot_v_j - v_i_dot_v_j**2
            if np.abs(denom) < 1e-8:
                continue
                
            p_diff = p_i - p_j
            p_diff_dot_v_i = np.dot(p_diff, v_i)
            p_diff_dot_v_j = np.dot(p_diff, v_j)
            
            s = (v_i_dot_v_j * p_diff_dot_v_j - v_j_dot_v_j * p_diff_dot_v_i) / denom
            t = (v_i_dot_v_i * p_diff_dot_v_j - v_i_dot_v_j * p_diff_dot_v_i) / denom
            
            if s <= 0 or t <= 0:
                # 視線の逆方向に交点がある場合はスキップ
                continue
                
            q_i = p_i + s * v_i
            q_j = p_j + t * v_j
            
            c_ij = (q_i + q_j) / 2.0
            intersections.append(c_ij)
            contrib_pairs.append((i, j))

    if return_pairs:
        return intersections, contrib_pairs
    return intersections


def visualize_gaze_rays_and_intersections(gaze_rays: list[dict], intersections: list[np.ndarray], ray_length: float = 10.0, best_point: np.ndarray = None):
    """gaze_rays (直線) と疑似交点 (点) を 3D 空間で可視化する"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. 直線 (Rays) の描画
        cam_origins = []
        for i, ray in enumerate(gaze_rays):
            origin = ray["origin"]
            direction = ray["direction"]
            cam_origins.append(origin)
            
            end_point = origin + direction * ray_length
            ax.plot([origin[0], end_point[0]], 
                    [origin[1], end_point[1]], 
                    [origin[2], end_point[2]], 
                    label=f"Ray {i} (Cam {ray['camera_id']})")
        
        # 2. カメラ位置をプロット
        cam_origins = np.array(cam_origins)
        if len(cam_origins) > 0:
            ax.scatter(cam_origins[:, 0], cam_origins[:, 1], cam_origins[:, 2], 
                       color='red', s=50, label='Camera Centers')
            
        # 3. 疑似交点 (intersections) をプロット
        intersections_arr = np.array(intersections)
        if len(intersections_arr) > 0:
            ax.scatter(intersections_arr[:, 0], intersections_arr[:, 1], intersections_arr[:, 2], 
                       color='blue', marker='x', s=100, label='Pseudo Intersections')
            
        # 4. 最大密度点 (best_point) をプロット
        if best_point is not None:
            ax.scatter(best_point[0], best_point[1], best_point[2], 
                       color='gold', marker='*', s=250, edgecolor='black', label='Highest Density Point', zorder=10)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Gaze Rays & Pseudo Intersections')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Warning: matplotlib is not installed. Skipping 3D visualization.")


def find_highest_density_point(
    points: list[np.ndarray], 
    R: float
) -> tuple[np.ndarray | None, list[int]]:

    n = len(points)
    if n == 0:
        return None, []

    densities = np.zeros(n)
    neighborhoods = []

    # 1. 各点 c_i について局所密度 rho_i と近傍集合 S_i を計算
    for i in range(n):
        c_i = points[i]
        S_i = []
        rho_i = 0.0
        
        for j in range(n):
            c_j = points[j]
            dist = np.linalg.norm(c_i - c_j)
            if dist <= R:
                S_i.append(j)
                rho_i += (1.0 - dist / R)
                
        densities[i] = rho_i
        neighborhoods.append(S_i)

    # 2. 密度が最大となる点を特定
    best_idx = np.argmax(densities)
    best_point = points[best_idx]
    best_neighborhood = neighborhoods[best_idx]

    return best_point, best_neighborhood
