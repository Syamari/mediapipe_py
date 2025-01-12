import cv2
import mediapipe as mp
import numpy as np
from utils import CvFpsCalc
import os


# MediaPipeのセットアップ
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ウェブカメラの準備
cap = cv2.VideoCapture(0)
# カメラの解像度を設定
use_camera_background = False


# カメラ背景のオンオフの関数；上記のグローばる変数で演算子を反転させる関数
def toggle_background():
    global use_camera_background
    use_camera_background = not use_camera_background


def capture_frame(background):
    # imageディレクトリがない場合でも作成しとける用に↓
    if not os.path.exists("image"):
        os.makedirs("image")

    # ファイル名を番号順に作成、（例: "image/capture_番号.png"）
    count_image = len(os.listdir("image"))
    filename = f"image/capture_{count_image}.png"

    # OpenCVで背景画像を保存
    cv2.imwrite(filename, background)
    print(f"Saved image: {filename}")


def draw_face(results, background):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=background,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
            )


def draw_hands(results2, background):
    if results2.multi_hand_landmarks:
        for hand_landmarks in results2.multi_hand_landmarks:
            # print('Handedness:', results.multi_handedness)
            mp_drawing.draw_landmarks(
                image=background,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
            )


def main():
    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # FaceMeshの設定
    # リソースを使い終わったら自動的にリリースできるようにコンテキストマネージャを使用する
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        # while True:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # 前回のフレームの時間を記録する変数
            prev_frame_time = 0

            # BGR画像をRGBに変換
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 検出を実行
            results = face_mesh.process(rgb_image)
            results2 = hands.process(rgb_image)

            # 背景の画像を作成
            height, width, _ = image.shape
            # 背景を画像かカメラで切り替えて設定
            if use_camera_background:
                background = image.copy()
            else:
                # background = np.ones((height, width, 3), np.uint8) * 255
                background = np.zeros((height, width, 3), np.uint8)

            # 結果を背景の画像上にface描画
            # faceを描画
            draw_face(results, background)
            # handsを描画
            draw_hands(results2, background)

            # FPSを表示（読み込んだモジュールを使用）
            display_fps = cvFpsCalc.get()
            cv2.putText(
                background,
                f"FPS: {int(display_fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # 画像を表示
            cv2.imshow("MediaPipe FaceMesh on Black Background", background)

            # キー入力受付の設定
            key = cv2.waitKey(10) & 0xFF
            # "q"を押すと終了
            if key == ord("q"):
                break
            # "s"を押すと画像を保存
            if key == ord("s"):
                capture_frame(background)
            # "b"を押すと背景を切り替え
            if key == ord("b"):
                toggle_background()


main()

cap.release()
cv2.destroyAllWindows()
