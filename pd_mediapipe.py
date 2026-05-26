from typing import Type

import cv2
import numpy as np
import mediapipe.python.solutions.pose as mpPose
import mediapipe.python.solutions.drawing_utils as mpDrawingUtils

class PD:

    pose_detector = None

    # 画像ごとに推論モデルを呼ぶとメモリを圧迫するため、クラス変数として保持
    @staticmethod
    def init_pose_detector():
        if PD.pose_detector is None:
            PD.pose_detector = mpPose.Pose(static_image_mode=True)

    @staticmethod
    def convert_BGR2RGB(img_BGR) -> np.array:
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        return img_RGB

    @staticmethod
    def convert_RGB2BGR(img_RGB) -> np.array:
        img_BGR = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2RGB)
        return img_BGR

    def __init__(self, img):
        PD.init_pose_detector()
        self.img = PD.convert_BGR2RGB(img)
        self.img_h = img.shape[0]
        self.img_w = img.shape[1]
        self.detection_result = self.detect_pose()

    def detect_pose(self) -> Type:
        detection_result = PD.pose_detector.process(self.img)
        return detection_result

    def is_pose_detected(self) -> bool:
        if self.detection_result.pose_landmarks is None:
            return False
        boundingbox_coordinate = self.get_boundingbox_coordinates()
        minXY, maxXY = boundingbox_coordinate[0], boundingbox_coordinate[1]
        if minXY[0] < 0 or minXY[1] < 0 or maxXY[0] < 0 or maxXY[1] < 0:
            return False
        return True
    
    # 任意のランドマーク座標を取得
    def get_landmark_coordinate(self, pose_num:int=0):
        landmark = self.detection_result.pose_landmarks.landmark[pose_num]
        x = int(landmark.x * self.img_w)
        y = int(landmark.y * self.img_h)
        return (x, y)

    def get_normalized_landmark_coordinates(self):
        landmarks = self.detection_result.pose_landmarks.landmark
        XCoordinates = np.array([landmarkCoordinates.x for landmarkCoordinates in landmarks])
        YCoordinates = np.array([landmarkCoordinates.y for landmarkCoordinates in landmarks])
        minXY = np.array([min(XCoordinates), min(YCoordinates)])
        maxXY = np.array([max(XCoordinates), max(YCoordinates)])
        size = maxXY - minXY
        size[size == 0] = 1.0  # ゼロ除算防止
        # 各キーポイントを正規化
        normalized = []
        for lm in landmarks:
            point = np.array([lm.x, lm.y])
            norm_point = (point - minXY) / size
            normalized.append(norm_point)
        return np.array(normalized)  # shape: (N, 2)

    def get_boundingbox_coordinates(self) -> list[tuple[float,float]]:
        landmarks = self.detection_result.pose_landmarks.landmark
        normalizedXCoordinates = np.array([landmarkCoordinates.x for landmarkCoordinates in landmarks])
        normalizedYCoordinates = np.array([landmarkCoordinates.y for landmarkCoordinates in landmarks])
        denormalizedXCoordinates = normalizedXCoordinates * self.img_w
        denormalizedYCoordinates = normalizedYCoordinates * self.img_h
        minXY = (int(min(denormalizedXCoordinates)), int(min(denormalizedYCoordinates)))
        maxXY = (int(max(denormalizedXCoordinates)), int(max(denormalizedYCoordinates)))
        BoundingBoxcCoordinates = [minXY, maxXY]
        return BoundingBoxcCoordinates   
    
    def get_boudingbox_surface(self):
        minXY, maxXY = self.get_boundingbox_coordinates()
        boundingbox_width = maxXY[0] - minXY[0]
        boundingbox_height = maxXY[1] - minXY[1]
        boundingbox_surface = boundingbox_width * boundingbox_height
        return boundingbox_surface

    def draw_pose_landmarks(self) -> np.array:
        img = self.img
        mpDrawingUtils.draw_landmarks(img, self.detection_result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        drawn_img = PD.convert_RGB2BGR(img)
        return drawn_img

    def draw_pose_landmark(self, pose_num:int=0, color=(255, 0, 0), thickness=5):
        landmark = self.detection_result.pose_landmarks.landmark[pose_num]
        x = int(landmark.x * self.img_w)
        y = int(landmark.y * self.img_h)
        if 0<=x<self.img_w and 0<=y<self.img_h:
            img = self.img
            cv2.circle(img, (x, y), thickness, color, -1)  # 5ピクセルの赤丸を描画
            drawn_img = PD.convert_RGB2BGR(img)
            return drawn_img
        else:
            print(f"Warning: Pose point ({x}, {y}) is out of image bounds.")
            return None

    def draw_boundingbox(self, color=(0, 255, 0), thickness=2) -> np.array:
        boundingbox_coordinate = self.get_boundingbox_coordinates()
        minXY, maxXY = boundingbox_coordinate[0], boundingbox_coordinate[1]
        img = self.img
        cv2.rectangle(img, minXY, maxXY, color, thickness)
        DrawnImg = PD.convert_RGB2BGR(img)
        return DrawnImg 

    def draw_pose_gazepoint_bb(self, gazepoint_coordinate, gazepoint_color=(255, 0, 0), bb_color=(0, 255, 0) ,thickness=2):
        img = self.img
        mpDrawingUtils.draw_landmarks(img, self.detection_result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        
        boundingbox_coordinate = self.get_boundingbox_coordinates()
        minXY, maxXY = boundingbox_coordinate[0], boundingbox_coordinate[1]
        cv2.rectangle(img, minXY, maxXY, bb_color, thickness)
        
        x, y = gazepoint_coordinate
        cv2.circle(img, (x, y), thickness+3, gazepoint_color, -1)
        DrawnImg = PD.convert_RGB2BGR(img)
        return DrawnImg
    
    def crop_boundingbox(self):
        boundingbox_coordinate = self.get_boundingbox_coordinates()
        minXY, maxXY = boundingbox_coordinate[0], boundingbox_coordinate[1]
        img = self.img
        x1, y1 = minXY
        x2, y2 = maxXY
        cropped_img = img[y1:y2, x1:x2]  
        return PD.convert_RGB2BGR(cropped_img)

    def output_detection_result(self):
        landmarks = self.detection_result.pose_landmarks.landmark
        for landmark_num, landmark_coordinates in enumerate(landmarks):
            landmarkName = mpPose.PoseLandmark(landmark_num).name
            print(f"{landmarkName:<20}:(x, y)=({landmark_coordinates.x:.3f},{landmark_coordinates.y:.3f})")


if __name__=="__main__":

    import sys
    import os

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    input_img = cv2.imread(input_path)
    os.makedirs(output_path, exist_ok=True)

    pose_detector = PD(input_img)
    if pose_detector.is_pose_detected():
        cv2.imwrite(os.path.join(output_path, 'detection_result.png'), pose_detector.draw_pose_landmarks())