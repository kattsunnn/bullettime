from .types import CropRecord, GazeRay
import numpy as np
from omni_directional_img_utils.e2p import E2P

def filter_pairs_by_camera_id(
    person_pairs: list[tuple[str, str, float]], 
    crop_records: list[CropRecord]
) -> list[tuple[str, str, float]]:
    # crop_recordsとの照合用マッピング
    path_to_record = {record.crop_img_path: record for record in crop_records}

    valid_pairs = []
    for path_i, path_j, similarity in person_pairs:
        record_i = path_to_record.get(path_i)
        record_j = path_to_record.get(path_j)
        if not record_i or not record_j:
            continue

        # camera_idが同じである場合は飛ばして次のペアを参照
        if record_i.camera_id == record_j.camera_id:
            continue

        valid_pairs.append((path_i, path_j, similarity))

    return valid_pairs

def remove_records_by_paths(
    crop_records: list[CropRecord],
    paths_to_remove: list[str]
) -> list[CropRecord]:
    """指定されたパスに対応するレコードをcrop_recordsから除外します"""
    remove_set = set(paths_to_remove)
    return [record for record in crop_records if record.crop_img_path not in remove_set]

def calculate_gaze_rays(crop_records: list[CropRecord], extrinsics_data: np.ndarray) -> list[GazeRay]:
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
        ray = GazeRay(
            camera_id=cam_idx,
            origin=camera_center,
            direction=direction,
            path=record.crop_img_path
        )
        gaze_rays.append(ray)
        
        print(f"Record {i} (Cam {cam_idx}): origin = {camera_center}, direction = {direction}")
        
    return gaze_rays

def find_rays_within_distance(
    ref_point: list[float] | np.ndarray,
    exclude_camera_ids: list[int],
    dist_th: float,
    extrinsics_data: np.ndarray,
    gaze_rays: list[GazeRay]
) -> list[GazeRay]:

    p = np.array(ref_point)
    close_rays = []
    skip_camera_ids = set(exclude_camera_ids)
    
    for ray in gaze_rays:
        # 除外処理
        if ray.camera_id in skip_camera_ids: # 復元に使用したカメラIDと同じものは除外
            continue
        cam_idx = ray.camera_id # 該当するカメラが存在しない場合
        if cam_idx >= len(extrinsics_data):
            continue

        # 世界座標系におけるz方向の光軸ベクトルを取得
        extrinsic = extrinsics_data[cam_idx]
        R = extrinsic[:3, :]
        z_vector = R.T[:, 2]
            
        C = ray.origin      # カメラの投影中心
        v = ray.direction   # 正規化された方向ベクトル
        
        # 注視点がカメラの後方に位置する場合はスキップ
        if np.dot(p - C, z_vector) < 0:
            continue
            
        # 点と直線の距離の計算
        d = np.linalg.norm(np.cross(p - C, v))
        
        if d <= dist_th:
            close_rays.append(ray)
            
    return close_rays

def visualize_geometric_reid(
    most_similar_rays: list[GazeRay],
    close_rays: list[GazeRay],
    ref_point: list[float] | np.ndarray,
    final_gaze_point: list[float] | np.ndarray,
    ray_length: float = 10.0
):
    """
    most_similar_pair の直線、close_rays の直線、ref_point の3次元点、
    および最終的な gaze_3d_point を 3D 空間で可視化します。
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. most_similar_rays (最初のペアの直線) の描画
        cam_origins_similar = []
        for i, ray in enumerate(most_similar_rays):
            origin = ray.origin
            direction = ray.direction
            cam_origins_similar.append(origin)
            
            end_point = origin + direction * ray_length
            ax.plot([origin[0], end_point[0]], 
                    [origin[1], end_point[1]], 
                    [origin[2], end_point[2]], 
                    color='blue', linestyle='--',
                    label=f"Initial Ray (Cam {ray.camera_id})" if i == 0 else "")
            
        # 2. close_rays (追加の直線) の描画
        cam_origins_close = []
        for i, ray in enumerate(close_rays):
            origin = ray.origin
            direction = ray.direction
            cam_origins_close.append(origin)
            
            end_point = origin + direction * ray_length
            ax.plot([origin[0], end_point[0]], 
                    [origin[1], end_point[1]], 
                    [origin[2], end_point[2]], 
                    color='orange',
                    label=f"Additional Ray (Cam {ray.camera_id})" if i == 0 else "")
            
        # カメラ位置をプロット
        if cam_origins_similar:
            cam_origins_similar = np.array(cam_origins_similar)
            ax.scatter(cam_origins_similar[:, 0], cam_origins_similar[:, 1], cam_origins_similar[:, 2], 
                       color='blue', marker='o', s=50, label='Initial Cameras')
        if cam_origins_close:
            cam_origins_close = np.array(cam_origins_close)
            ax.scatter(cam_origins_close[:, 0], cam_origins_close[:, 1], cam_origins_close[:, 2], 
                       color='orange', marker='o', s=50, label='Additional Cameras')

        # 3. ref_point (最初の復元された点) をプロット
        ref_pt = np.array(ref_point)
        ax.scatter(ref_pt[0], ref_pt[1], ref_pt[2], 
                   color='red', marker='x', s=150, label='Ref Point (Initial)')
        
        # 4. final_gaze_point (最終的な注視点) をプロット
        final_pt = np.array(final_gaze_point)
        ax.scatter(final_pt[0], final_pt[1], final_pt[2], 
                   color='gold', marker='*', s=250, edgecolor='black', label='Final Gaze Point')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Geometric Person Re-ID Visualization')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Warning: matplotlib is not installed. Skipping 3D visualization.")