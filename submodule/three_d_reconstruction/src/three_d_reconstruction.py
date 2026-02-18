import numpy as np
import submodule.omni_directional_img_utils as omni

def extrinsic_to_R_t(extrinsic):
    if extrinsic.shape != (4, 3):
        raise ValueError("camera_param must be of shape (4, 3)") 
   
    R = extrinsic[:3, :]
    t = extrinsic[3, :]

    return R, t

def uv_to_unit_sphere(u, v, src_w, src_h):
    e2p = omni.E2P(src_w, src_h)
    angle_u_deg, angle_v_deg = e2p.uv_to_angle(u, v)
    X, Y, Z = e2p.angle_to_unit_sphere(angle_u_deg, angle_v_deg)
    return X, Y, Z 

def xc_to_xw(xc, R, t):
    return R.T @ ( xc - t )

def xw_to_xc(xw, R, t):
    return R @ xw + t

def normalize_vec(vec):
    norm = np.linalg.norm(vec)
    normalized_vec = vec / norm
    return normalized_vec

def reconstruct_3d_points_from_omni_directional_img(extrinsics, corr_points_array, src_w, src_h):

    R_list = []
    t_list = [] 
    for extrinsic in extrinsics:
        R, t = extrinsic_to_R_t(extrinsic)
        R_list.append(R)
        t_list.append(t)
    R_array = np.array(R_list) 
    t_array = np.array(t_list)
   
    x0_array = np.array([ -R.T @ t for R, t in zip(R_array, t_array)])

    xc_list = []
    for corr_points in corr_points_array:
        xc_k = np.array([uv_to_unit_sphere(u, v, src_w, src_h) for u, v in corr_points ])
        xc_list.append(xc_k)
    xc_array = np.array(xc_list)

    xw_list = []
    for xc_k, R, t in zip(xc_array, R_array, t_array):
        xw_k = np.array([xc_to_xw(xc, R, t) for xc in xc_k])
        xw_list.append(xw_k)
    xw_array = np.array(xw_list)
    
    v_list = []
    for xw_k, x0_k in zip(xw_array, x0_array):
        v_k = xw_k -x0_k # ブロードキャスト
        v_k_normalized = np.array([normalize_vec(vec) for vec in v_k]) 
        v_list.append(v_k_normalized)
    v_array = np.array(v_list)

    P_list = []
    for v_k_normalized in v_array:
        P_k = np.array([np.identity(3) - v.reshape(3,1) @ v.reshape(1,3) for v in v_k_normalized])
        P_list.append(P_k)
    P_array = np.array(P_list)

    x_list = []
    num_of_alpha = corr_points_array.shape[1]
    for alpha in range(num_of_alpha):
        A = P_array[:, alpha].reshape(-1, 3)
        b = np.array([P_k @ x0_k for P_k, x0_k in zip(P_array[:, alpha], x0_array)]).reshape(-1)
        x_alpha = np.linalg.inv( A.T @ A ) @ A.T @ b
        x_list.append(x_alpha)
    x_array = np.array(x_list)

    return x_array
