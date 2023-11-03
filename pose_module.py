import time

import cv2
import mediapipe as mp
import pafy


class PoseDetector:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose()

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if draw and self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for _id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([_id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lm_list


def main():
    url = "https://www.youtube.com/watch?v=MP4A3cLuBP0"
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")

    cap = cv2.VideoCapture(best.url)
    p_time = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lm_list = detector.find_position(img)
        print(lm_list)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("image", img)

        cv2.waitKey(1)


if __name__ == '__main__':
    main()
