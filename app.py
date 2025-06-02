import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# 🧠 YOLO + MediaPipe 초기화
yolo_model = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose

# 📐 각도 계산 함수
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# 💬 팔 피드백 함수
def feedback_arm(arm_angle):
    if 160 <= arm_angle <= 180:
        return "Good arm angle!"
    elif 140 <= arm_angle < 160:
        return "Arm not fully extended."
    else:
        return "Bad arm extension!"

# 📹 Streamlit WebRTC Video Transformer
class PoseVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # MediaPipe Pose
        results_pose = self.pose.process(img_rgb)

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            h, w, _ = img.shape

            right_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            right_elbow = (int(landmarks[14].x * w), int(landmarks[14].y * h))
            right_wrist = (int(landmarks[16].x * w), int(landmarks[16].y * h))

            # 팔 각도 계산
            arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            feedback = feedback_arm(arm_angle)

            # 시각화
            points = [right_shoulder, right_elbow, right_wrist]
            for point in points:
                cv2.circle(img, point, 5, (255, 0, 0), -1)

            cv2.line(img, right_shoulder, right_elbow, (0, 255, 0), 2)
            cv2.line(img, right_elbow, right_wrist, (0, 255, 0), 2)

            # 피드백 텍스트
            cv2.putText(img, f"Arm Angle: {int(arm_angle)} deg", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, feedback, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return img

# 🎯 Streamlit 페이지
st.title("🏐 AI Spike Pose Analysis")
st.write("Real-time arm angle analysis for spike posture!")

webrtc_streamer(key="spike-pose-analysis", video_transformer_factory=PoseVideoTransformer)
