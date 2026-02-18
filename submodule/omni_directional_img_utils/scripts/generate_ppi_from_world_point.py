import camera_pose_estimation.camera_calibration_utils as ccu
import img_utils.img_utils as iu
from e2p import E2P
import numpy as np
import sys

img_dir = sys.argv[1]
extrinsics_dir = sys.argv[2]
output_dir = sys.argv[3]
ref_distance = sys.argv[4]

imgs = iu.load_imgs(img_dir)

extrinsics_paths = iu.glob_file_paths_from_dir(extrinsics_dir, "*extrinsics*")

world_point = np.array([-1.56960260e-01,  7.30445014e+00, 1.48442004e+00])

ppis = []
for img, extrinsics_path in zip(imgs, extrinsics_paths):

    r, t = ccu.load_extrinsics_file(extrinsics_path)
    camera_point = ccu.xw_to_xc(world_point, r, t)
    angle_u_deg, angle_v_deg = E2P.eye_vec_to_angle(camera_point)
    e2p = E2P(img.shape[1], img.shape[0])

    camera_distance = np.linalg.norm(camera_point) 
    scale = float(camera_distance) / float(ref_distance)

    e2p.generate_map(30, 30, angle_u_deg, angle_v_deg, 0, scale)
    ppi = e2p.generate_img(img)
    ppis.append(ppi)

iu.save_imgs(ppis, output_dir)