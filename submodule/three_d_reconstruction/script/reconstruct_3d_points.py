import sys
import json
from pathlib import Path
import numpy as np
from three_d_reconstruction import reconstruct_3d_points_from_omni_directional_img 

corr_points_json_path = sys.argv[1]
camera_params_dir_path = sys.argv[2]
src_w = 3840
src_h = 1920 

# 対応点の取得
with open(corr_points_json_path, 'r', encoding='utf-8') as f:
    corr_points_array =  np.array(json.load(f))
# print(corr_points_array.shape)

# カメラパラメータの取得
camera_param_paths = Path(camera_params_dir_path) 
extrinsics = []
for camera_param_path in list(camera_param_paths.iterdir()):
    camera_param = np.loadtxt(camera_param_path.resolve())
    extrinsic = camera_param[:4]
    extrinsics.append(extrinsic)

x_array = reconstruct_3d_points_from_omni_directional_img(extrinsics, corr_points_array, src_w, src_h )
print(x_array)

# サンプル実行例
# python .\demo.py .\sample\corr_points.json .\sample\camera_params\