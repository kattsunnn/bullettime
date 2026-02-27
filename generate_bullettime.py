from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import pdb
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

from submodule import omni_directional_img_utils as omni
from submodule import three_d_reconstruction as tdr 
from submodule import camera_calibration as camcal

from pd import PD
from img_utils import img_utils as iu
from person_re_identification import osnet_reid as reid 
_map_cache = {}

def get_or_create_map(src_img_w, src_img_h, fov_w_deg, fov_h_deg, angle_u_deg, angle_v_deg, scale=1):
    cache_key = (src_img_w, src_img_h, fov_w_deg, fov_h_deg, angle_u_deg, angle_v_deg, scale)
    
    if cache_key not in _map_cache:
        map = omni.E2P(src_img_w, src_img_h)
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
            ppi = omni.PPI(img_e, ppi, theta_eye, phi_eye)
            ppis.append(ppi)
    return ppis

# 透視投影画像を生成
def generate_ppi(img_e, theta_eye, phi_eye, fov, scale=1):
    img_e_w = img_e.shape[1]
    img_e_h = img_e.shape[0]
    map = get_or_create_map(img_e_w, img_e_h, fov, fov, theta_eye, phi_eye, scale)
    ppi = map.generate_img(img_e)
    return omni.PPI(img_e, ppi, theta_eye, phi_eye)

def generate_gazed_ppi(ppi, fov):
    pose_detector = PD(ppi.get_ppi())
    if pose_detector.is_pose_detected():
        landmark_coordinate_x, landmark_coordinate_y = pose_detector.get_landmark_coordinate()
        theta_e, phi_e = ppi.convert_ppi_point_to_angle_coor(landmark_coordinate_x, landmark_coordinate_y)
        gazed_ppi = generate_ppi(ppi.get_src_img(), theta_e, phi_e, fov)
        return gazed_ppi

# スケールから適切なFOVを計算。
def calc_optimal_fov_from_scale(ppi, scale):
    focal_length = ppi.get_focal_length()
    w = ppi.get_ppi().shape[1]
    s_p = scale
    fov_rad = np.arctan2(w, (focal_length*s_p))
    fov_deg = np.rad2deg(fov_rad)
    return fov_deg

# Todo: 入力を画像に修正
def get_gaze_point_on_omni(ppi, gaze_point_num=0):
    pose_detector_for_ppi = PD(ppi.get_ppi())
    if pose_detector_for_ppi.is_pose_detected():
        # 注視点（鼻）の座標を取得
        landmark_coordinate_x, landmark_coordinate_y = pose_detector_for_ppi.get_landmark_coordinate(gaze_point_num)
        # 注視点の全方位画像上の角度座標を取得
        theta_e, phi_e = ppi.convert_ppi_point_to_angle_coor(landmark_coordinate_x, landmark_coordinate_y)
    return theta_e, phi_e

def grouping_points(points, eps=5, min_samples=1):
    if not isinstance(points, np.ndarray):
        raise TypeError("pointsがNumPy配列ではありません。np.ndarrayを渡してください。")
    if points.size == 0:
        return None
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # グルーピングしたラベル配列を出力。例：[0, 0, 1, 1, -1]（ラベルが -1 の場合はノイズ）
    labels = dbscan.fit_predict(points)
    # 点をグループ分けした辞書を作成。例：｛　0: array([[x1, y1], [x2, y2]])　｝
    grouped_points = {}
    for label in set(labels):
        grouped_points[label] = points[labels == label]
    return grouped_points

def centering_points(points):
    if points.shape[0] == 0:
        return None
    centered_point = np.mean(points, axis=0)
    return centered_point

def is_gaze_img(img, threshold=0.05):
    pose_detector = PD(img)
    if not pose_detector.is_pose_detected():
        return False
    
    gaze_point = np.array([pose_detector.get_landmark_coordinate()])
    center = np.array([img.shape[1] / 2, img.shape[0] / 2])
    # 対角長を使って距離を計算
    dist = np.linalg.norm(gaze_point - center)
    diag = np.linalg.norm(np.array([img.shape[1], img.shape[0]]))
    normalized_dist = dist / diag
    if normalized_dist > threshold:
        return False
    return True

# Todo: FOVが低下した場合はスケールからスケーリングを行う
# 人が透視投影画像に1/kの高さで写る。例： k=2のとき、人が透視投影画像の1/2の高さになる
def scaling_person_by_height(ppi, fov, k=2):
    pose_detector_for_adjusted = PD(ppi.get_ppi())
    if pose_detector_for_adjusted.is_pose_detected():
        minXY, maxXY = pose_detector_for_adjusted.get_boundingbox_coordinates()
        H_b = maxXY[1] - minXY[1]
        H_i = ppi.get_ppi().shape[0]
        ppi_scale = H_i/(k * H_b)

        theta_e, phi_e = ppi.get_gaze_point_of_angle_coor()

        # スケールに応じて解像度を自動調節
        src_img = ppi.get_src_img()
        scaling_fov = calc_optimal_fov_from_scale(ppi, ppi_scale)
        # 解像度が低下する場合は変更しない
        if scaling_fov > fov: 
            # 画角を変えると解像度も変わる
            scaled_ppi = generate_ppi(  src_img, 
                                        theta_e,
                                        phi_e,
                                        scaling_fov)
        else:
            # スケールを変えると解像度は変わらない
            scaled_ppi = generate_ppi(  src_img, 
                                        theta_e,
                                        phi_e,
                                        fov,
                                        scale=ppi_scale) 
        return scaled_ppi
    
def detect_and_draw_pose(img):
    pose_detector = PD(img)
    if pose_detector.is_pose_detected():
        img_pose = pose_detector.draw_pose_landmarks()
        return img_pose

def filter_none(lst):
    return [ elem for elem in lst if elem is not None ]

def generate_scaled_gazed_ppis(img, fov, output_path, file_name_pattern):
    # 正面方向の透視投影画像を生成
    ppis = generate_front_ppis(img, fov)
    # デバッグ
    if ppis:
        ppis_raw = [ppi.get_ppi() for ppi in ppis]
        iu.save_imgs(ppis_raw, f"{output_path}/00_ppi", f"{file_name_pattern}_{{}}")
        ppis_pose = filter_none([detect_and_draw_pose(img) for img in ppis_raw])
        if ppis_pose: iu.save_imgs(ppis_pose, f"{output_path}/00_ppi_pose", f"{file_name_pattern}_{{}}")

    # 注視画像の生成と注視画像チェック
    gazed_ppis = filter_none([generate_gazed_ppi(ppi, fov) for ppi in ppis ])
    checked_gazed_ppis = [ gazed_ppi for gazed_ppi in gazed_ppis if is_gaze_img(gazed_ppi.get_ppi())]
    if not checked_gazed_ppis: return None
    # デバック
    checked_gazed_ppis_raw = [ checked_gazed_ppi.get_ppi() for checked_gazed_ppi in checked_gazed_ppis]        
    iu.save_imgs(checked_gazed_ppis_raw, f"{output_path}/01_checked_gazed_ppi", f"{file_name_pattern}_{{}}")

    # 注視画像を生成
    gaze_point_candidates = filter_none([get_gaze_point_on_omni(checked_gazed_ppi) for checked_gazed_ppi in checked_gazed_ppis])
    if not gaze_point_candidates: return None
    grouped_points = grouping_points(np.array(gaze_point_candidates))
    centered_points = [centering_points(grouped_point) for grouped_point in grouped_points.values()]
    grouped_ppis = [ generate_ppi(img, cp[0], cp[1], fov) for cp in centered_points ]
    # デバッグ
    grouped_ppis_raw = [ grouped_ppi.get_ppi() for grouped_ppi in grouped_ppis ]
    iu.save_imgs(grouped_ppis_raw , f"{output_path}/02_grouped_ppi", f"{file_name_pattern}_{{}}")
    grouped_ppis_pose = filter_none([detect_and_draw_pose(img) for img in grouped_ppis_raw ])
    if grouped_ppis_pose: iu.save_imgs(grouped_ppis_pose, f"{output_path}/02_grouped_ppi_pose", f"{file_name_pattern}_{{}}")

    # スケーリングと注視画像チェック
    scaled_ppis = filter_none([scaling_person_by_height(grouped_ppi, fov) for grouped_ppi in grouped_ppis])
    scaled_ppis = [ scaled_ppi for scaled_ppi in scaled_ppis if is_gaze_img(scaled_ppi.get_ppi()) ]
    if not scaled_ppis: return None
    # デバッグ
    scaled_ppis_raw = [ scaled_ppi.get_ppi() for scaled_ppi in scaled_ppis ]
    iu.save_imgs(scaled_ppis_raw, f"{output_path}/03_scaled_ppi", f"{file_name_pattern}_{{}}")
    scaled_ppis_pose = filter_none([detect_and_draw_pose(img) for img in scaled_ppis_raw])
    if scaled_ppis_pose: iu.save_imgs(scaled_ppis_pose, f"{output_path}/03_scaled_ppi_pose", f"{file_name_pattern}_{{}}")

    return scaled_ppis

def crop_person(img):
    pose_detector = PD(img)
    if pose_detector.is_pose_detected():
        return pose_detector.crop_boundingbox()

@dataclass
class PPIWithExtrinsics:
    ppi: omni.PPI
    extrinsics: np.ndarray

# Todo: スケールをFovで統一するか検討する
# def calc_scaled_fov(ref_fov, ref_d, target_d):
#     target_fov = 2 * np.arctan( np.tan(ref_fov/2) * (ref_d/target_d)) 
#     return target_fov

def generate_ppi_from_world_point(world_point, extrinsics, img_e, fov, scale_distance=5):
    r = extrinsics[:3, :]
    t = extrinsics[3, :]
    gaze_vec = camcal.xw_to_xc(world_point, r, t)    
    theta_e, phi_e = omni.E2P.eye_vec_to_angle(gaze_vec)
    cam_distance = np.linalg.norm(gaze_vec)
    scale = cam_distance/scale_distance
    generator = omni.E2P(img_e.shape[1], img_e.shape[0])
    generator.generate_map(fov, fov, theta_e, phi_e, 0, scale)
    ppi = generator.generate_img(img_e)
    return ppi


def generate_bullettime(imgs, fov, extrinsics, output_path):
    scaled_gazed_ppis_list = [generate_scaled_gazed_ppis(img, fov, output_path, f"camera_{idx:02d}") for idx, img in enumerate(imgs)]
    # 部分画像の生成
    cropped_img_map = {}
    for cam_idx, scaled_gaze_ppis in enumerate(scaled_gazed_ppis_list):
        if scaled_gaze_ppis is None: continue
        for num_idx, scaled_gaze_ppi in enumerate(scaled_gaze_ppis):
            cropped_img = crop_person(scaled_gaze_ppi.get_ppi())
            file_name = f"camera_{cam_idx:02d}_{num_idx:02d}.jpg"
            output_dir = Path(output_path) / "04_cropped"
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path =  output_dir / file_name
            file_path_str = str(file_path)
            cropped_img_map[file_path_str] = PPIWithExtrinsics(scaled_gaze_ppi, extrinsics[cam_idx])
            cv2.imwrite(file_path_str, cropped_img)

    cropped_img_paths = list(cropped_img_map.keys()) 
    #Person ReId
    person_cluster = reid.OSNetReID.cluster_imgs(cropped_img_paths, min_samples=2)

    gaze_points_of_world_coor = []
    src_img_w = imgs[0].shape[1]
    src_img_h = imgs[0].shape[0]

    for label, paths in person_cluster.items():
        print(label)
        for path in paths:
            print(path)
        if label < 0 or len(paths) < 2:
            continue
        corr_points = [ [cropped_img_map[path].ppi.get_gaze_point_of_img_coor()] 
                        for path in paths   ]
        corr_points_array = np.array(corr_points)
        extrinsics_list = [ cropped_img_map[path].extrinsics
                            for path in paths ]
        three_d_gaze_point = tdr.reconstruct_3d_points_from_omni_directional_img(
            extrinsics_list, 
            corr_points_array, 
            src_img_w, 
            src_img_h   )
        gaze_points_of_world_coor.append(three_d_gaze_point.flatten())
    
    for idx, gaze_point in enumerate(gaze_points_of_world_coor): 
        print(gaze_point)
        bullettime = [ generate_ppi_from_world_point(gaze_point, extrinsics, img_e, 60, 3) for img_e, extrinsics in zip(imgs, extrinsics)]
        iu.save_imgs(bullettime, f"{output_path}/05_bullettime", f"camera_{idx:02d}_{{}}")


# Todo: 全体を分割する
if __name__ == '__main__':

    import glob
    import os
    import sys

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    extrinsics_dir = sys.argv[3]
    fov = float(sys.argv[4])

    imgs = iu.load_imgs(input_dir)
    extrinsics_list = []
    extrinsics_paths = iu.glob_file_paths_from_dir(extrinsics_dir, "*extrinsics*")
    for extrinsics_path in extrinsics_paths:
        extrinsics = np.loadtxt(extrinsics_path)
        extrinsics_list.append(extrinsics)

    generate_bullettime(imgs, fov, extrinsics_list, output_dir)