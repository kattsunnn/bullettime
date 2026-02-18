# Todo: 180方向で画像に欠損が生じる問題
# Todo: 引数にValidateを追加
# Todo: 変数名等をPPIと統一

import math
import numpy as np
import cv2

class E2P:
    def __init__(self, src_img_w, src_img_h):

        # 全方位画像の大きさ
        self.src_img_w = src_img_w
        self.src_img_h = src_img_h
        # 焦点距離
        self.f = self.src_img_w / (2*np.pi)

        self.map_u = None
        self.map_v = None

    #画角から適切な出力画像の横サイズを計算
    def calc_optimal_width(self, fov_w_deg):
        fov_w_rad = np.deg2rad(fov_w_deg)
        optimal_w = int((self.src_img_w / np.pi) * np.tan(fov_w_rad / 2.0))
        return optimal_w
    
    # 画角から適切な出力画像の縦サイズを計算
    def calc_optimal_height(self, fov_h_deg):
        fov_h_rad = np.deg2rad(fov_h_deg)
        optimal_h = int(2.0 * np.tan(fov_h_rad / 2.0) * (self.src_img_h / np.pi))
        return optimal_h

    def calc_optimal_fov_w(self, dst_width):
        fov_w_rad = 2 * np.arctan2((dst_width/self.src_img_w) * np.pi)
        fov_w_deg = np.rad2deg(fov_w_rad)
        return fov_w_deg

    def calc_optimal_fov_h(self, dst_height):
        fov_h_rad = 2 * np.arctan2((dst_height/self.src_img_h) * (np.pi/2))
        fov_h_deg = np.rad2deg(fov_h_rad)
        return fov_h_deg

    def angle_to_uv(self, angle_u_deg, angle_v_deg):
        angle_u_rad = np.deg2rad(angle_u_deg)
        angle_v_rad = np.deg2rad(angle_v_deg)
        u = (angle_u_rad + np.pi) * (self.src_img_w/(2*np.pi))
        v = (angle_v_rad + (np.pi/2)) * (self.src_img_h/np.pi)
        return u, v

    def uv_to_angle(self, u, v):
        angle_u_rad = (u - (self.src_img_w/2)) * ((2*np.pi)/self.src_img_w) 
        angle_v_rad = (v - (self.src_img_h/2)) * ((np.pi)/self.src_img_h) 
        angle_u_deg = np.rad2deg(angle_u_rad)
        angle_v_deg = np.rad2deg(angle_v_rad)
        return angle_u_deg, angle_v_deg
    
    @staticmethod
    def eye_vec_to_angle(eye_vec):
        angle_u_rad = np.arctan2(eye_vec[0], eye_vec[2])
        angle_v_rad = np.arctan2(eye_vec[1], np.sqrt(eye_vec[0]**2 + eye_vec[2]**2))
        angle_u_deg = np.rad2deg(angle_u_rad)
        angle_v_deg = np.rad2deg(angle_v_rad)
        return angle_u_deg, angle_v_deg
        
    @staticmethod
    def angle_to_unit_sphere(angle_u_deg, angle_v_deg):
        angle_u_rad = np.deg2rad(angle_u_deg)
        angle_v_rad = np.deg2rad(angle_v_deg)
        X = np.cos(angle_v_rad) * np.sin(angle_u_rad)
        Y = np.sin(angle_v_rad)
        Z = np.cos(angle_v_rad) * np.cos(angle_u_rad)
        return X, Y, Z
    
    @staticmethod
    def angle_to_r_mat(angle_u_deg, angle_v_deg):
        angle_u_rad = np.deg2rad(angle_u_deg)
        angle_v_rad = np.deg2rad(angle_v_deg)
        r_mat = np.dot(E2P.rotation_y(angle_u_rad), E2P.rotation_x(angle_v_rad))
        return r_mat
    
    @staticmethod
    def r_mat_to_angle(r_mat):
        optical_axis = np.array([0,0,1], dtype=np.float32)
        rotated_optical_axis = r_mat @ optical_axis
        angle_u_rad = np.arctan2(rotated_optical_axis[0], rotated_optical_axis[2])
        angle_v_rad = np.arctan2(rotated_optical_axis[1], np.sqrt(rotated_optical_axis[0]**2 + rotated_optical_axis[2]**2))
        angle_u_deg = np.rad2deg(angle_u_rad)
        angle_v_deg = np.rad2deg(angle_v_rad)
        return angle_u_deg, angle_v_deg

    # X軸周りの回転行列
    @staticmethod
    def rotation_x(angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[1.0, 0.0, 0.0],
                    [0.0, cos_a, sin_a],
                      [0.0, -sin_a, cos_a]], dtype=np.float64)
        return R

    # Y軸周りの回転行列
    @staticmethod
    def rotation_y(angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[cos_a, 0.0, sin_a],
                      [0.0, 1.0, 0.0],
                      [-sin_a, 0.0, cos_a]], dtype=np.float64)
        return R

    # 任意の軸周りの回転行列
    @staticmethod
    def rotation_by_axis(n, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        l_cos_a = 1.0 - cos_a
        R = np.array([[cos_a + n[0] * n[0] * l_cos_a,
                       n[0] * n[1] * l_cos_a - n[2] * sin_a,
                       n[2] * n[0] * l_cos_a + n[1] * sin_a],
                      [n[0] * n[1] * l_cos_a + n[2] * sin_a,
                       cos_a + n[1] * n[1] * l_cos_a,
                       n[1] * n[2] * l_cos_a - n[0] * sin_a],
                      [n[2] * n[0] * l_cos_a - n[1] * sin_a,
                       n[1] * n[2] * l_cos_a + n[0] * sin_a,
                       cos_a + n[2] * n[2] * l_cos_a]], dtype=np.float64)
        return R
    
    # 画像生成用のマップ作成関数
    def generate_map(self,
                     fov_w_deg,
                     fov_h_deg,
                     angle_u_deg,
                     angle_v_deg,
                     angle_z_deg,
                     scale=1):
        angle_u_rad = np.deg2rad(angle_u_deg)
        angle_v_rad = np.deg2rad(angle_v_deg)
        angle_z_rad = np.deg2rad(angle_z_deg)

        dst_w = self.calc_optimal_width(fov_w_deg)
        dst_h = self.calc_optimal_height(fov_h_deg)

        # 回転行列の計算
        R = np.dot(self.rotation_y(angle_u_rad), self.rotation_x(angle_v_rad))
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        Ri = self.rotation_by_axis(np.dot(R, axis), angle_z_rad)
        R = np.dot(Ri, R)
        
        # 一括計算用の出力画像の画素座標データ
        u = np.arange(0, dst_w, 1)
        v = np.arange(0, dst_h, 1)
        dst_u, dst_v = np.meshgrid(u, v)

        # 視線ベクトルを計算
        x = dst_u - dst_w * 0.5
        y = dst_v - dst_h * 0.5
        z = self.f * scale * np.ones((dst_h, dst_w))
        
        # print(x[dst_h_h, dst_w_h], y[dst_h_h, dst_w_h], z[dst_h_h, dst_w_h])
        # 回転行列で回転
        Xx = R[0][0] * x + R[0][1] * y + R[0][2] * z
        Xy = R[1][0] * x + R[1][1] * y + R[1][2] * z
        Xz = R[2][0] * x + R[2][1] * y + R[2][2] * z

        # 視線ベクトルから角度を計算
        theta = np.arctan2(Xx, Xz)
        phi = np.arctan2(Xy, np.sqrt(Xx**2 + Xz**2))

        # 角度から入力画像の座標を計算
        self.map_u = ((theta + np.pi) * (self.src_img_w / (2*np.pi))).astype(np.float32)
        self.map_v = ((phi + (np.pi/2)) * (self.src_img_h / np.pi)).astype(np.float32)

    # 画像生成
    def generate_img(self, src_img):
        if self.map_u is None or self.map_v is None:
            raise Exception("Error: Map is not generated yet. Please call generate_map() first.")
        return cv2.remap(src_img, self.map_u, self.map_v, cv2.INTER_LINEAR)

#Todo: Outputの整理
if __name__ == '__main__':


    import sys
    from ppi import PPI

    src_img = cv2.imread(sys.argv[1]) # 全方位画像
    fov_w_deg = float(sys.argv[2])
    fov_h_deg = float(sys.argv[3])
    angle_u_deg = float(sys.argv[4])
    angle_v_deg = float(sys.argv[5])
    angle_z_deg = float(sys.argv[6])

    e2p = E2P(src_img.shape[1], src_img.shape[0])
    e2p.generate_map( fov_w_deg, fov_h_deg, 
                      angle_u_deg, angle_v_deg, angle_z_deg,
                      scale=1.0)
    dst_img = e2p.generate_img(src_img)

    ppi = PPI(src_img, dst_img, angle_u_deg, angle_v_deg)
    print(ppi.get_gaze_point_of_angle_coor())
    print(ppi.get_gaze_point_of_img_coor())
    print(ppi.convert_ppi_point_to_angle_coor(ppi.get_ppi().shape[1]/2, ppi.get_ppi().shape[0]/2))
    cv2.imshow("dst", dst_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # from ppi import PPI

    # ppi = PPI(src_img, dst_img, angle_u_deg, angle_v_deg)
    # print(ppi.get_angular_coordinate(dst_img.shape[1]/2, dst_img.shape[0]/2))
    # # 画像表示

    # e2p = E2P(src_img.shape[1], src_img.shape[0])
    # e2p.generate_map( fov_w_deg, fov_h_deg, 
    #                   angle_u_deg, angle_v_deg, angle_z_deg,
    #                   scale=1.0)
    # dst_img = e2p.generate_img(src_img)

    # output_path = sys.argv[7]
    # cv2.imwrite(output_path, dst_img)