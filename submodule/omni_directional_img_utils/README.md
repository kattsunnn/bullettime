# omni_directional_img_utils
全方位画像に関する機能を提供するライブラリ

## e2p.py
全方位画像のサイズを元に、任意の透視投影画像のマップを生成するクラス

> ### __init__(self, src_img_w, src_img_h):
コンストラクタ

**引数：**
- **src_img_w:** 全方位画像の横サイズ
- **src_img_h:** 全方位画像の縦サイズ

> ### calc_optimal_width(self, fov_w_deg):
画角から適切な透視投影画像の横サイズを計算

> ### calc_optimal_height(self, fov_h_deg):
画角から適切な透視投影画像の縦サイズを計算

> ### calc_optimal_fov_w(self, dst_width):
透視投影画像の横サイズから適切な画角（横）を計算

> ### calc_optimal_fov_h(self, dst_height):
透視投影画像の横サイズから適切な画角（縦）を計算

> ### angle_to_uv(self, angle_u_deg, angle_v_deg):
全方位画像の角度座標を画像座標に変換

> ### uv_to_angle(self, u, v):
全方位画像の画像座標を角度座標に変換


> ### generate_map(self, fov_w_deg, fov_h_deg, angle_u_deg, angle_v_deg, angle_z_deg, scale=1):
透視投影画像のmapを作成

#### 引数：
- **fov_w_deg:** 透視投影画像の画角（横）
- **fov_h_deg:** 透視投影画像の画角（縦）
- **angle_u_deg:** 透視投影画像の視線方向（横）。全方位画像の角度座標として指定
- **angle_v_deg:** 透視投影画像の視線方向（縦）。全方位画像の角度座標として指定
- **angle_z_deg:** 透視投影画像の視線の回転。回転角度を指定
- **scale:** 透視投影画像のスケール

#### 戻り値：
- なし

> ### generate_img(self, src_img):
generate_mapで作成したmapに従い透視投影画像を作成

## ppi.py
透視投影画像を保持するクラス

> ### __init__(self, img_e, img_p, angle_u, angle_v):
コンストラクタ

**引数：**
- **img_e:** 全方位画像のnp配列
- **img_p:** 透視投影画像のnp配列
- **angle_u:** 透視投影画像生成に使用した角度座標　（横）
- **angle_v:** 透視投影画像生成に使用した角度座標　（縦）


> ### get_angular_coordinate(self, u_p, v_p):
透視投影画像の座標から全方位画像の角度座標を得る関数

**引数：**
- **u_v:** 透視投影画像の画像座標（横）
- **v_p:** 透視投影画像の画像座標（縦）

**戻り値：** 
- **angle_u_deg:** 透視投影画像の画像座標に対応した全方位画像の角度座標（横）
- **angle_v_deg:** 透視投影画像の画像座標に対応した全方位画像の角度座標（横）


## generate_ppi_gui.py
GUIで全方位画像の座標を指定して、透視投影画像を作成する

> ### generate_ppi_gui(src_img, scale=1, fov_w_deg=30, fov_v_deg=30)
GUIで全方位画像の座標を指定して、透視投影画像を作成する

**引数：**
- **src_img:** cv2.imreadで読み込んだ全方位画像のnp配列
- **scale:** GUIで画像を表示する際のスケール
- **fov_w_deg:** 作成する透視投影画像の画角（横）
- **fov_h_deg:** 作成する透視投影画像の画角（縦）

**戻り値：** 
- **ppi:** 透視投影画像のnp配列