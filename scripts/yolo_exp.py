import numpy as np
import os
import json

import img_utils as iu
from omni_directional_img_utils.e2p import E2P
from omni_directional_img_utils.ppi import PPI  
# from camera_calibration.camera_calibration_utils import xw_to_xc
# from person_re_identification.osnet import OSNet
# from three_d_reconstruction import reconstruct_3d_points_from_omni_directional_img 

from pd_yolo_pose import PD_YOLO

_map_cache = {}

def get_or_create_map(src_img_w, src_img_h, fov_w_deg, fov_h_deg, angle_u_deg, angle_v_deg, scale=1):
    cache_key = (src_img_w, src_img_h, fov_w_deg, fov_h_deg, angle_u_deg, angle_v_deg, scale)
    
    if cache_key not in _map_cache:
        map = E2P(src_img_w, src_img_h)
        map.generate_map(fov_w_deg, fov_h_deg, angle_u_deg, angle_v_deg, 0, scale)
        _map_cache[cache_key] = map
    
    return _map_cache[cache_key]

# ToDo: 透視投影画像生成するマップは１回の計算にできそう
# 透視投影画像群の生成
def generate_front_ppis(img_e, fov, overlap=0.5):
    # 使用する全方位カメラの解像度
    img_e_w = img_e.shape[1]
    img_e_h = img_e.shape[0]
    # 透視投影画像を生成する視線角度の設定
    THETA_RANGE = 90
    PHI_RANGE = 60
    angle_step = fov * (1 - overlap)
    first_step = fov/2
    theta_eyes = np.arange(-THETA_RANGE + first_step, THETA_RANGE, angle_step)
    phi_eyes = np.arange(PHI_RANGE - first_step, -PHI_RANGE, -angle_step)

    ppis = []

    for j, phi_eye in enumerate(phi_eyes):
        for i, theta_eye in enumerate(theta_eyes):
            # 透視投影画像の生成
            map = get_or_create_map(img_e_w, img_e_h, fov, fov, theta_eye, phi_eye)
            ppi = map.generate_img(img_e)
            ppi = PPI(img_e, ppi, theta_eye, phi_eye)
            ppis.append(ppi)
    return ppis

def get_gaze_points_on_omni(ppis, gaze_points_dict):
    gaze_points_on_omni = []
    for idx, ppi in enumerate(ppis):
        if idx not in gaze_points_dict:
            continue
        gaze_points = gaze_points_dict[idx]
        gaze_points = gaze_points.reshape(-1,3)
        converted_gaze_points = [
            [*ppi.convert_ppi_point_to_angle_coor(pt[0], pt[1]), pt[2]]
            for pt in gaze_points
        ]
        gaze_points_on_omni.extend(converted_gaze_points)
    return np.array(gaze_points_on_omni).reshape(-1, 3)

def distance_based_nms(points: np.ndarray, dist_th: float = 5.0) -> np.ndarray | None:
    if points is None or len(points) == 0:
        return None 
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"入力配列は (N, 3) の形状である必要があります。現在の形状: {points.shape}")
    if dist_th < 0:
        raise ValueError(f"距離の閾値 (dist_th) には 0 以上の数値を指定してください。指定値: {dist_th}")

    sorted_indices = points[:, 2].argsort()[::-1]
    candidates_points = points[sorted_indices]
    selected_points  = []
    while len(candidates_points) > 0:
        p_max = candidates_points[0]
        selected_points.append(p_max)
        if len(candidates_points) == 1:
            break
        remaining_points = candidates_points[1:]
        diff = remaining_points[:, :2] - p_max[:2]
        distances = np.linalg.norm(diff, axis=1)
        keep_mask = distances >= dist_th
        candidates_points = remaining_points[keep_mask]
    return np.array(selected_points) 

def generate_ppi(img_e, theta_eye, phi_eye, fov, scale=1):
    img_e_w = img_e.shape[1]
    img_e_h = img_e.shape[0]
    map = get_or_create_map(img_e_w, img_e_h, fov, fov, theta_eye, phi_eye, scale)
    ppi = map.generate_img(img_e)
    return PPI(img_e, ppi, theta_eye, phi_eye)

def generate_gazed_ppis(img_e, fov, output_path, file_name_pattern, input_size: int = 640):
    # 正面方向の透視投影画像を生成
    ppis = generate_front_ppis(img_e, fov, overlap=0.2)
    ppis_raw = [ppi.get_ppi() for ppi in ppis] 
    iu.save_imgs(ppis_raw, f"{output_path}/00_ppi", f"{file_name_pattern}_{{}}")

    pose_detector = PD_YOLO(input_size=input_size) 
    _ = pose_detector.detect_pose(ppis_raw)
    plotted_ppis = pose_detector.plot_detected_poses()
    iu.save_imgs(plotted_ppis, f"{output_path}/01_plotted_ppi", f"{file_name_pattern}_{{}}")

    gaze_points_dict = pose_detector.get_landmark_points()
    gaze_points_on_omni = get_gaze_points_on_omni(ppis, gaze_points_dict) 
    clusterd_gaze_points = distance_based_nms(gaze_points_on_omni)
    save_data = {
            "ppi": [arr.tolist() for arr in gaze_points_dict.values()],
            "omni": [arr.tolist() for arr in gaze_points_on_omni],
            "clustered": [arr.tolist() for arr in clusterd_gaze_points]
        }
    os.makedirs(f"{output_path}/02_data", exist_ok=True)
    save_file_path = f"{output_path}/02_data/{file_name_pattern}_data.json"
    with open(save_file_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
    print(f"Text data successfully saved to: {save_file_path}")

    gaze_ppis =  [ generate_ppi(img_e, pt[0], pt[1], fov) for pt in clusterd_gaze_points ]
    gaze_ppis_raw = [ ppi.get_ppi() for ppi in gaze_ppis ] 
    iu.save_imgs(gaze_ppis_raw, f"{output_path}/03_gaze_ppi", f"{file_name_pattern}_{{}}")
# def crop_person(img):
#     pose_detector = PD(img)
#     if pose_detector.is_pose_detected():
#         return pose_detector.crop_boundingbox()

# @dataclass
# class PPIWithExtrinsics:
#     ppi: PPI
#     extrinsics: np.ndarray

# def generate_ppi_from_world_point(world_point, extrinsics, img_e, fov, scale_distance=5):
#     r = extrinsics[:3, :]
#     t = extrinsics[3, :]
#     gaze_vec = xw_to_xc(world_point, r, t)    
#     theta_e, phi_e = E2P.gaze_vec_to_angle(gaze_vec)
#     cam_distance = np.linalg.norm(gaze_vec)
#     scale = cam_distance/scale_distance
#     generator = E2P(img_e.shape[1], img_e.shape[0])
#     generator.generate_map(fov, fov, theta_e, phi_e, 0, scale)
#     ppi = generator.generate_img(img_e)
#     return ppi


# def generate_bullettime(imgs, fov, extrinsics, output_path):
#     # 注視画像の生成
#     scaled_gazed_ppis_list = [generate_scaled_gazed_ppis(img, fov, output_path, f"camera_{idx:02d}") for idx, img in enumerate(imgs)]
#     # 部分画像の生成
#     cropped_img_map = {}
#     for cam_idx, scaled_gaze_ppis in enumerate(scaled_gazed_ppis_list):
#         if scaled_gaze_ppis is None: continue
#         for num_idx, scaled_gaze_ppi in enumerate(scaled_gaze_ppis):
#             cropped_img = crop_person(scaled_gaze_ppi.get_ppi())
#             file_name = f"camera_{cam_idx:02d}_{num_idx:02d}.jpg"
#             output_dir = Path(output_path) / "04_cropped"
#             output_dir.mkdir(parents=True, exist_ok=True)
#             file_path =  output_dir / file_name
#             file_path_str = str(file_path)
#             cropped_img_map[file_path_str] = PPIWithExtrinsics(scaled_gaze_ppi, extrinsics[cam_idx])
#             cv2.imwrite(file_path_str, cropped_img)

#     #Person ReId
#     cropped_img_paths = list(cropped_img_map.keys()) 
#     person_cluster, _ = OSNet.cluster_imgs_with_auto_eps(cropped_img_paths, min_samples=2)

#     # 3次元復元
#     gaze_points_of_world_coor = []
#     src_img_w = imgs[0].shape[1]
#     src_img_h = imgs[0].shape[0]

#     log = []

#     for label, paths in person_cluster.items():
#         log.append(str(label))
#         for path in paths:
#             log.append(path)
#         if label < 0 or len(paths) < 2:
#             continue
#         corr_points = [ [cropped_img_map[path].ppi.get_gaze_point_of_img_coor()] 
#                         for path in paths   ]
#         corr_points_array = np.array(corr_points)
#         extrinsics_list = [ cropped_img_map[path].extrinsics
#                             for path in paths ]
#         three_d_gaze_point = reconstruct_3d_points_from_omni_directional_img(
#             extrinsics_list, 
#             corr_points_array, 
#             src_img_w, 
#             src_img_h   )
#         gaze_points_of_world_coor.append(three_d_gaze_point.flatten())
    
#     # バレットタイム映像生成
#     for idx, gaze_point in enumerate(gaze_points_of_world_coor): 
#         log.append(str(gaze_point))
#         bullettime = [ generate_ppi_from_world_point(gaze_point, extrinsics, img_e, 60, 3) for img_e, extrinsics in zip(imgs, extrinsics)]
#         iu.save_imgs(bullettime, f"{output_path}/05_bullettime", f"camera_{idx:02d}_{{}}")

#     with open(f"{output_path}/log.txt", "w", encoding="utf-8") as f:
#         f.write("\n".join(log) + "\n")

# Todo: 全体を分割する
if __name__ == '__main__':

    import glob
    import os
    import sys

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # extrinsics_dir = sys.argv[3]
    fov = float(sys.argv[3])
    input_size = int(sys.argv[4])

    imgs = iu.load_imgs(input_dir)
    # extrinsics_list = []
    # extrinsics_paths = iu.glob_file_paths(extrinsics_dir, "*extrinsics*")
    # for extrinsics_path in extrinsics_paths:
    #     extrinsics = np.loadtxt(extrinsics_path)
    #     extrinsics_list.append(extrinsics)
    for idx, img in enumerate(imgs):
        file_name_pattern = f"camera_{idx:02d}"
        generate_gazed_ppis(img, fov, output_dir, file_name_pattern, input_size=input_size)