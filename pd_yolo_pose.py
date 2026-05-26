from ultralytics import YOLO
import numpy as np

class PD_YOLO:
    def __init__(self, model_path: str = 'yolo26x-pose.pt', input_size: int = 640, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.input_size = input_size
        self.conf = conf
        self.results = None
        self.img = None

    def detect_pose(self, img):
        self.img = img
        self.results = self.model(img, imgsz=self.input_size, conf=self.conf, verbose=False)
        return self.results

    # def _get_detected_results(self):
    #     if self.results is None:
    #         return
    #     for result in self.results:
    #         if len(result.keypoints.data) > 0:
    #             yield result
                 
    # 指定したキーポイントインデックスの座標を収集 
    def get_landmark_points(self, keypoint_idx: int = 0) -> list:
        target_points = []
        for result in self.results:
            people_kps = result.keypoints.data
            if len(people_kps) == 0:
                continue
            else:
                target_kps = people_kps[:, keypoint_idx, :].cpu().numpy()
                target_points.append(target_kps)
        return np.vstack(target_points)

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
    pose_detector = PD_YOLO(input_size=1960)
    results = pose_detector.detect_pose(input_path)
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

