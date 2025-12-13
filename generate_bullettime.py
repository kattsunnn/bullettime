import pdb
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

from pd import PD
from ppi import PPI
import search_similar_img as ssi
import img_utils.img_utils as iu
import super_resolution.super_resolution as sr
from person_re_identification.osnet_reid import OSNetReID
from omni_directional_img_utils.e2p import E2P

FOV = 30
_map_cache = {}

def get_or_create_map(src_img_w, src_img_h, fov, theta_eye, phi_eye):
    cache_key = (src_img_w, src_img_h, fov, theta_eye, phi_eye)
    
    if cache_key not in _map_cache:
        map = E2P(src_img_w, src_img_h)
        dst_w = map.calc_optimal_width(fov)
        dst_h = map.calc_optimal_height(fov)
        map.generate_map(dst_w, dst_h, theta_eye, phi_eye, 0)
        _map_cache[cache_key] = map
    
    return _map_cache[cache_key]

# ToDo: 透視投影画像生成するマップは１回の計算にできそう
# 透視投影画像群の生成
def generate_front_ppis(img_e, fov=FOV, overlap=0.5):
    # 使用する全方位カメラの解像度
    src_img_h = img_e.shape[0]
    src_img_w = img_e.shape[1]
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
            map = get_or_create_map(src_img_w, src_img_h, fov, theta_eye, phi_eye)
            dst_img = map.generate_img(img_e)
            # 保存
            ppi = PPI(img_e, dst_img, theta_eye, phi_eye)
            ppis.append(ppi)
    return ppis

# 透視投影画像を生成
def generate_ppi(img_e, theta_eye, phi_eye, scale=1, fov=FOV):
    img_e_w = img_e.shape[1]
    img_e_h = img_e.shape[0]
    map = E2P(img_e_w, img_e_h)
    dst_w = map.calc_optimal_width(fov)
    dst_h = map.calc_optimal_height(fov)
    map.generate_map(dst_w, dst_h, theta_eye, phi_eye, 0, scale)
    ppi = map.generate_img(img_e)
    return PPI(img_e, ppi, theta_eye, phi_eye)

# スケールから適切なFOVを計算。
def calc_optimal_fov_from_scale(ppi, scale):
    focal_length = ppi.get_focal_length()
    w = ppi.get_ppi_w()
    s_p = scale
    fov_rad = np.arctan2(w, (focal_length*s_p))
    fov_deg = np.rad2deg(fov_rad)
    return fov_deg

# Todo: 入力を画像に修正
def collect_gaze_point_candidates(ppis, gaze_point_num=0):
    gaze_point_candidates = []
    for ppi in ppis:
        pose_detector_for_ppi = PD(ppi.get_ppi())
        if pose_detector_for_ppi.is_pose_detected():
            # 注視点（鼻）の座標を取得
            landmark_coordinate_x, landmark_coordinate_y = pose_detector_for_ppi.get_landmark_coordinate(gaze_point_num)
            # 注視点の全方位画像上の角度座標を取得
            theta_e, phi_e = ppi.get_angular_coordinate(landmark_coordinate_x, landmark_coordinate_y)
            gaze_point_candidates.append([theta_e, phi_e])
    return np.array(gaze_point_candidates, dtype=float)

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
def scaling_person_by_height(ppi, k=2):
    pose_detector_for_adjusted = PD(ppi.get_ppi())
    if pose_detector_for_adjusted.is_pose_detected():
        minXY, maxXY = pose_detector_for_adjusted.get_boundingbox_coordinates()
        H_b = maxXY[1] - minXY[1]
        H_i = ppi.get_ppi_h()
        ppi_scale = H_i/(k * H_b)

        # スケールに応じて解像度を自動調節
        scaling_fov = calc_optimal_fov_from_scale(ppi, ppi_scale)
        # 解像度が低下する場合は変更しない
        if scaling_fov < FOV: 
            scaled_ppi = generate_ppi(  ppi.get_src_img(), 
                                        ppi.get_angle_u(),
                                        ppi.get_angle_v(),
                                        fov= scaling_fov)
        else:
            # 解像度を一定に保ちたい場合。
            scaled_ppi = generate_ppi(  ppi.get_src_img(), 
                                        ppi.get_angle_u(),
                                        ppi.get_angle_v(),
                                        scale=ppi_scale) 
        return scaled_ppi
    
# 人が透視投影画像に1/kの面積で写る。例： k=2のとき、人が透視投影画像の1/2の面積になる
def scaling_person_by_surface(ppi, k=2):
    pose_detector_for_adjusted = PD(ppi.get_ppi())
    if pose_detector_for_adjusted.is_pose_detected():
        bb_S = pose_detector_for_adjusted.get_boudingbox_surface()
        img_ratio = ppi.get_ppi_w() / ppi.get_ppi_h()
        W_dash = np.sqrt(bb_S * img_ratio * k)
        img_w = ppi.get_ppi_w()
        ppi_scale = img_w / W_dash
        scaled_ppi = generate_ppi(  ppi.get_src_img(), 
                                    ppi.get_angle_u(),
                                    ppi.get_angle_v(),
                                    ppi_scale)
        return scaled_ppi

def detect_and_draw_pose(img):
    pose_detector = PD(img)
    if pose_detector.is_pose_detected():
        img_pose = pose_detector.draw_pose_landmarks()
        return img_pose

def filter_none(lst):
    return [ elem for elem in lst if elem is not None ]

def generate_scaled_gaze_imgs(img, output_path, file_name_pattern):
    # 正面方向の透視投影画像を生成
    ppis = generate_front_ppis(img)
    # デバッグ
    ppis_raw = [ppi.get_ppi() for ppi in ppis]
    if ppis_raw: iu.save_imgs(ppis_raw, f"{output_path}/00_ppi", f"{file_name_pattern}_{{}}")
    ppis_pose = filter_none(map(detect_and_draw_pose, ppis_raw))
    if ppis_pose: iu.save_imgs(ppis_pose, f"{output_path}/01_ppi_pose", f"{file_name_pattern}_{{}}")

    # model_path = "super_resolution/models/EDSR_x4.pb"
    # ppis = [ PPI(ppi.get_src_img(), sr.edsr_x4(ppi.get_ppi(), model_path), ppi.get_angle_u(), ppi.get_angle_v()) for ppi in ppis ]

    # 注視画像を生成
    collect_gaze_point_candidate = collect_gaze_point_candidates(ppis)
    if collect_gaze_point_candidate.size == 0: return None
    grouped_points = grouping_points(collect_gaze_point_candidate)
    centered_points = list(map(centering_points, grouped_points.values()))
    gaze_ppis = [ generate_ppi(img, cp[0], cp[1]-90) for cp in centered_points ]
    # デバッグ
    gaze_ppis_raw = [ gaze_ppi.get_ppi() for gaze_ppi in gaze_ppis ]
    gaze_ppis_pose = filter_none(map(detect_and_draw_pose, gaze_ppis_raw))
    if gaze_ppis_raw: iu.save_imgs(gaze_ppis_pose, f"{output_path}/02_gaze_ppi_pose", f"{file_name_pattern}_{{}}")

    # スケーリングと注視画像チェック
    scaled_ppis = filter_none(map(scaling_person_by_height, gaze_ppis))
    scaled_ppis = [ ppi for ppi in scaled_ppis if is_gaze_img(ppi.get_ppi()) ]
    if not scaled_ppis: return
    # デバッグ
    scaled_ppis_raw = [ scaled_ppi.get_ppi() for scaled_ppi in scaled_ppis ]
    scaled_ppis_pose = filter_none(map(detect_and_draw_pose, scaled_ppis_raw))
    if scaled_ppis_raw: iu.save_imgs(scaled_ppis_pose, f"{output_path}/03_scaled_ppi_pose", f"{file_name_pattern}_{{}}")

    return [ppi.get_ppi() for ppi in scaled_ppis]

def generate_crop_img(img):
    pose_detector = PD(img)
    if pose_detector.is_pose_detected():
        return pose_detector.crop_boundingbox()

#Todo:カメラ番号を渡して、画像を出力する
def generate_same_person_imgs(imgs, output_path):
    scaled_gaze_imgs_list = [generate_scaled_gaze_imgs(img, output_path, f"camera_{idx}") for idx, img in enumerate(imgs)]
    # デバック
    for idx, scaled_gaze_imgs in enumerate(scaled_gaze_imgs_list):
        if not scaled_gaze_imgs:
            continue
        cropped_gaze_imgs = filter_none(map(generate_crop_img, scaled_gaze_imgs))
        iu.save_imgs(cropped_gaze_imgs, f"{output_path}/04_cropped", f"camera_{idx}_{{}}")

    cropped_gaze_img_paths = iu.load_img_paths_from_dir(f"{output_path}/04_cropped")
    groups = OSNetReID.cluster_imgs(cropped_gaze_img_paths)
    for label, paths in groups.items():
        print(f"Cluster {label}:")
        for path in paths:
            print(f"  {path}")


# Todo: 全体を分割する
if __name__ == '__main__':

    import glob
    import os
    import sys

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    img_paths = iu.load_img_paths_from_dir(input_path)
    img_paths.sort()
    print(img_paths)
    imgs = [cv2.imread(img_path) for img_path in img_paths ]

    generate_same_person_imgs(imgs, output_path)