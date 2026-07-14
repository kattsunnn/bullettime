from ultralytics import YOLO
import numpy as np

class PD_YOLO:
    def __init__(self, model_path: str = 'yolo26x-pose.pt'):
        self.model = YOLO(model_path)
        self.results = None

    def detect_pose(self, imgs: list[np.ndarray], input_size: int = 640, pd_conf: float = 0.5):
        self.results = self.model(imgs, imgsz=input_size, conf=pd_conf, verbose=False)
        return self.results

    # 指定したキーポイントインデックスの座標を収集 
    def get_landmark_points(self, keypoint_idx: int = 0) -> dict:
        target_points_dict = {} 
        for idx, result in enumerate(self.results):
            people_kps = result.keypoints.data
            # キーポイントが存在すれば
            if len(people_kps) > 0:
                target_kps = people_kps[:, keypoint_idx, :].cpu().numpy() # keypoint_idxのkpsのみ抜き取る
                target_kps = target_kps.reshape(-1, 3) # ベクトルが2次元になることを保証
                target_points_dict[idx] = target_kps
        return target_points_dict

    # 検出した情報を画像にプロット
    def plot_detected_poses(self, 
                            show_conf: bool = True,
                            show_boxes: bool = True,
                            line_width: float = None,
                            kpt_radius: int = 5,):
        plotted_imgs = [] 
        for result in self.results:
            people_kps = result.keypoints.data
            if len(people_kps) == 0:
                continue
            else:
                plotted_img = result.plot(
                    boxes=show_boxes,
                    labels=show_boxes,
                    conf=show_conf,
                    line_width=line_width,
                    kpt_radius=kpt_radius
                )
                plotted_imgs.append(plotted_img)
        return plotted_imgs
    
# 検出した人物のバウンディングボックス画像を切り出す
    def get_cropped_people(self) -> dict:
        cropped_imgs_dict = {}

        for idx, result in enumerate(self.results):
            # 元の画像（Numpy配列）を取得
            orig_img = result.orig_img
            # 検出されたバウンディングボックスを取得
            boxes = result.boxes.xyxy.cpu().numpy()
            
            cropped_list = []
            for box in boxes:
                # 座標を整数(int)に変換（スライスには整数が必要なため）
                x1, y1, x2, y2 = map(int, box)
                # Numpyのスライス機能を使って画像を切り抜き: image[y_start:y_end, x_start:x_end]
                cropped_img = orig_img[y1:y2, x1:x2]
                cropped_list.append(cropped_img)
            cropped_imgs_dict[idx] = cropped_list
            
        return cropped_imgs_dict

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import cv2

    parser = argparse.ArgumentParser(
        description="YOLO-Poseで骨格検出を行うスクリプト"
    )

    parser.add_argument(
        "input_path", type=str, help="入力画像のパスを指定してください"
    ) 
    args = parser.parse_args()

    input_path = Path(args.input_path)
    pose_detector = PD_YOLO()
    results = pose_detector.detect_pose(input_path, input_size=2000)
    target_points = pose_detector.get_landmark_points()
    plotted_imgs = pose_detector.plot_detected_poses()
    for i, img in enumerate(plotted_imgs):
        # ウィンドウに画像を表示（第1引数はウィンドウのタイトル）
        cv2.imshow(f"Detected Pose - Image {i}", img)
    
        # 【重要】キーボードのいずれかのキーが押されるまで待機する
        # 引数の 0 は「キーが押されるまで無限に待つ」という意味です
        cv2.waitKey(0)

    # 3. 処理が終わったら、開いたすべてのウィンドウを綺麗に閉じる
    cv2.destroyAllWindows()
    breakpoint()

