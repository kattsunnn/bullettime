import numpy as np
from sklearn.cluster import DBSCAN
from three_d_reconstruction import reconstruct_3d_points_from_omni_directional_img
from .types import CropRecord, ReconstructionRecord

def validate_and_filter_clusters(
    person_cluster: dict[str, list[str]],
    all_crop_records: list[CropRecord],
    extrinsics: np.ndarray,
) -> dict[str, list[str]]:
    crop_record_map = {record.crop_img_path: record for record in all_crop_records}
    cleaned_cluster = {}
    for label, paths in person_cluster.items():
        try:
            int_label = int(label)
        except ValueError:
            int_label = -1
        # ノイズクラスタ（未分類）は除外する
        if int_label < 0:
            continue
        # カメラIDごとに関連するパスをグループ化
        camera_to_paths = {}
        for path in paths:
            record = crop_record_map.get(path)
            if record is None:
                continue
            cam_id = record.camera_id
            # カメラIDが外部パラメータの範囲内であるかも確認
            if cam_id >= len(extrinsics):
                continue
            if cam_id not in camera_to_paths:
                camera_to_paths[cam_id] = []
            camera_to_paths[cam_id].append(path)
        # 同一カメラIDから複数選ばれているパスを除外
        valid_paths = []
        for cam_id, path_list in camera_to_paths.items():
            if len(path_list) == 1:
                valid_paths.append(path_list[0])
            else:
                pass
        # 3次元復元に利用できない（残った画像が2個以下の）クラスタは削除する
        if len(valid_paths) >= 2:
            cleaned_cluster[label] = valid_paths
    return cleaned_cluster

def reconstruct_cluster_3d(
    paths: list[str],
    all_crop_records: list[CropRecord],
    extrinsics: np.ndarray,
    src_w: int,
    src_h: int,
) -> ReconstructionRecord:
    """1つの人物クラスタから対応点（注視点および長さ測定点）の3次元座標を復元する"""
    crop_record_map = {record.crop_img_path: record for record in all_crop_records}
    extrinsics_list = []
    corr_points = []
    camera_ids = []
    for path in paths:
        record = crop_record_map[path]
        camera_id = record.camera_id
        extrinsics_list.append(extrinsics[camera_id])
        # gaze_point_omni_uv (2,) と length_points_omni_uv (N, 2) を結合して (1+N, 2) の配列にする
        gaze_pt = record.gaze_point_omni_uv
        length_pts = record.length_points_omni_uv
        combined_pts = np.vstack([gaze_pt[np.newaxis, :], length_pts])
        
        corr_points.append(combined_pts)
        camera_ids.append(camera_id)
    # 複数対応点の3D復元を一回の呼び出しで実行
    three_d_points = reconstruct_3d_points_from_omni_directional_img(
        np.array(extrinsics_list),
        np.array(corr_points),
        src_w,
        src_h
    )
    # gaze_point (最初の一点) と length_points (残りの点) に分割
    three_d_gaze = three_d_points[0]
    three_d_length = three_d_points[1:]
    
    gaze_point_3d = three_d_gaze.tolist()
    length_points_3d = three_d_length.tolist()
    # length_points_3d の各点を順番につないだ3次元距離の合計値（全距離）を計算
    diffs = np.diff(three_d_length, axis=0)
    total_length = float(np.sum(np.linalg.norm(diffs, axis=-1)))
    return ReconstructionRecord(
        gaze_point_3d=gaze_point_3d,
        length_points_3d=length_points_3d,
        total_length=total_length,
        cameras=camera_ids,
        paths=paths
    )

def cluster_gaze_points(
    reconstruction_results: dict[str, ReconstructionRecord]
) -> list[list[float]]:
    """各人物の注視点(gaze_point_3d)をクラスタリングし、クラスタごとの代表値（平均値）のリストを返す"""
    if not reconstruction_results:
        return []
    # 各レコードから gaze_point_3d と total_length を抽出
    records = list(reconstruction_results.values())
    gaze_points = np.array([rec.gaze_point_3d for rec in records])
    total_lengths = [rec.total_length for rec in records]
    # total_lengthの平均値をDBSCANのepsに設定
    eps_val = float(np.mean(total_lengths))
    # DBSCANの実行 (min_samples=1)
    dbscan = DBSCAN(eps=eps_val, min_samples=1)
    labels = dbscan.fit_predict(gaze_points)
    unique_labels = set(labels)
    representative_points = []
    for label in unique_labels:
        # そのクラスタに属する点を抽出
        cluster_pts = gaze_points[labels == label]
        # 複数要素を持つクラスタは平均値を代表値とする
        rep_pt = np.mean(cluster_pts, axis=0)
        representative_points.append(rep_pt.tolist())
    return representative_points