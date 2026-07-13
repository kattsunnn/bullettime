import numpy as np

from omni_directional_img_utils.e2p import E2P
from omni_directional_img_utils.ppi import PPI  
from three_d_reconstruction import extrinsic_to_R_t, xw_to_xc

# 透視投影マップのキャッシュ
_map_cache = {}

# 透視投影画像のマップを作成．生成を効率化
def get_or_create_ppi_map(src_img_w, src_img_h, fov_w, fov_h, eye_w, eye_h, scale=1):
    cache_key = (src_img_w, src_img_h, fov_w, fov_h, eye_w, eye_h, scale)
    
    if cache_key not in _map_cache:
        map = E2P(src_img_w, src_img_h)
        map.generate_map(fov_w, fov_h, eye_w, eye_h, 0, scale)
        _map_cache[cache_key] = map
    
    return _map_cache[cache_key]

# 正面方向の透視投影画像群の生成
def generate_front_ppis(src_img, fov_w, fov_h=None, overlap=0.25, range_w=90, range_h=60):
    # 使用する全方位カメラの解像度
    src_img_w = src_img.shape[1]
    src_img_h = src_img.shape[0]
    if fov_h is None:
        fov_h = fov_w
    # 透視投影画像を生成する視線角度の設定
    step_w = fov_w * (1 - overlap)
    step_h = fov_h * (1 - overlap)
    first_step_w = fov_w / 2
    first_step_h = fov_h / 2
    eyes_w = np.arange(-range_w + first_step_w, range_w, step_w)
    eyes_h= np.arange(-range_h + first_step_h, range_h, step_h)

    ppis = []

    for eye_h in eyes_h:
        for eye_w in eyes_w:
            # 透視投影画像の生成
            map = get_or_create_ppi_map(src_img_w, src_img_h, fov_w, fov_h, eye_w, eye_h)
            ppi = map.generate_img(src_img)
            ppi = PPI(src_img, ppi, eye_w, eye_h)
            ppis.append(ppi)
    return ppis

def generate_ppi(src_img, fov_w, fov_h, eye_w, eye_h, scale=1):
    src_img_w = src_img.shape[1]
    src_img_h = src_img.shape[0]
    map = get_or_create_ppi_map(src_img_w, src_img_h, fov_w, fov_h, eye_w, eye_h, scale)
    ppi = map.generate_img(src_img)
    return PPI(src_img, ppi, eye_w, eye_h)

def generate_ppi_from_world_point(world_point: np.ndarray, extrinsic: np.ndarray, src_img: np.ndarray, fov_w: float, fov_h: float, scale_distance: float = 3.0) -> np.ndarray:
    R, t = extrinsic_to_R_t(extrinsic)
    gaze_vec = xw_to_xc(world_point, R, t)
    eye_w, eye_h = E2P.gaze_vec_to_angle(gaze_vec)
    cam_distance = np.linalg.norm(gaze_vec)
    scale = cam_distance / scale_distance
    generator = E2P(src_img.shape[1], src_img.shape[0])
    generator.generate_map(fov_w, fov_h, eye_w, eye_h, 0, scale)
    ppi = generator.generate_img(src_img)
    return ppi