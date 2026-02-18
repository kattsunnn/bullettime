import numpy as np
import img_utils.img_utils as iu
from e2p import E2P

def generate_ppi_with_gui(src_img, scale=1, fov_w_deg=30, fov_v_deg=30):
    point, _ = iu.get_single_point_with_gui(src_img, scale)
    u = point[0]
    v = point[1]
    ppi_generator = E2P(src_img.shape[1], src_img.shape[0])
    angle_u_deg, angle_v_deg = ppi_generator.uv_to_angle(u, v)
    ppi_generator.generate_map(fov_w_deg, fov_v_deg, angle_u_deg, angle_v_deg, 0)
    ppi = ppi_generator.generate_img(src_img)
    r_mat = E2P.angle_to_r_mat(angle_u_deg, angle_v_deg)
    return ppi, r_mat

def generate_ppi_from_uv(src_img, fov_w_deg, fov_h_deg, u, v):
    e2p = E2P(src_img.shape[1], src_img.shape[0])
    angle_u_deg, angle_v_deg = e2p.uv_to_angle(u, v)
    e2p.generate_map( fov_w_deg, fov_h_deg, angle_u_deg, angle_v_deg, 0, scale=1.0)
    ppi = e2p.generate_img(src_img)
    r_mat = E2P.calc_r_xy(angle_u_deg, angle_v_deg)
    return ppi, r_mat


# if __name__ == "__main__":

# サンプル：generate_ppi_from_uv
# 実行例：python .\generate_ppi.py .\sample\sample.jpg 30 30 .\sample\sample.txt 
    # img_path = sys.argv[1]
    # fov_w_deg = float(sys.argv[2])
    # fov_h_deg = float(sys.argv[3])
    # coordinates_txt_path = sys.argv[4]

    # # 画像読み込み
    # src_img = iu.load_imgs(img_path)
    # # 座標読み込み
    # coordinates = iu.load_coodinates_from_txt(coordinates_txt_path)

    # for u, v in coordinates:
    #     ppi = generate_ppi_from_uv(src_img, fov_w_deg, fov_h_deg, u, v)
    #     iu.show_imgs(ppi)