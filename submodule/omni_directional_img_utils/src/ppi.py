import math
import numpy as np
from .e2p import E2P

class PPI:
    def __init__(self, img_e, img_p, theta_e_deg, phi_e_deg):
        self.img_e = img_e
        self.img_e_h = img_e.shape[0]
        self.img_e_w = img_e.shape[1]

        self.img_p = img_p
        self.img_p_h = img_p.shape[0]
        self.img_p_w = img_p.shape[1]

        self.theta_e_deg = theta_e_deg 
        self.phi_e_deg = phi_e_deg

    def get_src_img(self):
        return self.img_e
    def get_ppi(self):
        return self.img_p
        
    def get_focal_length(self):
        return self.img_e_w / (2*np.pi)

    def get_gaze_point_of_angle_coor(self):
        return self.theta_e_deg, self.phi_e_deg
    def get_gaze_point_of_img_coor(self):
        theta_e_rad = np.deg2rad(self.theta_e_deg)
        phi_e_rad = np.deg2rad(self.phi_e_deg)
        u = (theta_e_rad + np.pi) * (self.img_e_w/(2*np.pi))
        v = (phi_e_rad + (np.pi/2)) * (self.img_e_h/np.pi)
        return u, v

    def convert_ppi_point_to_angle_coor(self, u_p, v_p):
        # 透視投影画像の画像座標から回転前の3次元視線ベクトルを計算
        focal_length = self.img_e_w/(2*np.pi)
        x = u_p-(self.img_p_w/2)
        y = v_p-(self.img_p_h/2)
        z = focal_length
        # 透視投影画像の生成に使用した回転行列を計算
        theta_e_rad = np.deg2rad(self.theta_e_deg)
        phi_e_rad = np.deg2rad(self.phi_e_deg)
        R = np.dot(E2P.rotation_y(theta_e_rad), E2P.rotation_x(phi_e_rad))
        # 回転後の視線ベクトルを計算
        rotated_x = R[0][0] * x + R[0][1] * y + R[0][2] * z
        rotated_y = R[1][0] * x + R[1][1] * y + R[1][2] * z
        rotated_z = R[2][0] * x + R[2][1] * y + R[2][2] * z
        # 回転した視線ベクトルから全方位画像の角度座標を計算
        theta_e, phi_e = E2P.eye_vec_to_angle([rotated_x, rotated_y, rotated_z])
        return theta_e, phi_e

    

    

