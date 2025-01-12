以下では、app.py のコードを具体的に説明します。

---

## 全体の役割

このプログラムは、OpenCV と MediaPipe を使ってカメラ動画を解析し、

1. 顔（FaceMesh）のランドマークを描画
2. 手（Hands）のランドマークを描画
3. FPS（フレーム毎秒）を表示

を同時に行うサンプルです。

FPS 計測用に、utils/cvfpscalc.py で定義されている

CvFpsCalc

クラスを使用しています。

---

## 詳細な構成

1. ### ライブラリのインポート
   - OpenCV:

cv2

- MediaPipe:

mediapipe as mp

- NumPy:

numpy as np

- ユーティリティ:

CvFpsCalc

これらをまとめて使うことで、カメラ画像の取得、画像処理、FPS 計測が行えます。

2. ### MediaPipe の初期化

   ```python
   mp_drawing = mp.solutions.drawing_utils
   mp_face_mesh = mp.solutions.face_mesh
   mp_hands = mp.solutions.hands

   hands = mp_hands.Hands(
       min_detection_confidence=0.7,
       min_tracking_confidence=0.5,
   )
   ```

   - MediaPipe の顔検出（FaceMesh）や手検出（Hands）で、閾値（信頼度）を設定しています。
   - 手の検出には [`Hands`](https://github.com/google/mediapipe/blob/master/mediapipe/examples/desktop/hand_tracking) が用いられ、検出や追跡の精度を

min_detection_confidence

と

min_tracking_confidence

で指定しています。

3. ### カメラの準備

   ```python
   cap = cv2.VideoCapture(0)
   ```

   - カメラ（Web カメラ）を起動します。0 はデフォルトのカメラを指定しています。

4. ### 顔を描画する関数:

draw_face

```python
def draw_face(results, background):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=background,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
            )
```

- MediaPipe FaceMesh が返す

results.multi_face_landmarks

をループで取り出しながら、

mp_drawing.draw_landmarks()

を使って背景画像 (

background

) 上に顔の特徴点を描画します。

-

FACEMESH_TESSELATION

は、顔全体の細かい接続線を描くための設定です。

5. ### 手を描画する関数:

draw_hands

```python
def draw_hands(results2, background):
    if results2.multi_hand_landmarks:
        for hand_landmarks in results2.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=background,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
            )
```

- MediaPipe Hands が返す

results2.multi_hand_landmarks

をループで取り出しながら、手のランドマークを描画します。

-

mp_hands.HAND_CONNECTIONS

を与えることで、指の骨格接続線が表現されます。

6. ### メイン関数:

main

```python
def main():
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # BGR -> RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 顔検出
            results = face_mesh.process(rgb_image)
            # 手検出
            results2 = hands.process(rgb_image)

            # 背景画像を生成: 今回は白一色
            height, width, _ = image.shape
            background = np.ones((height, width, 3), np.uint8) * 255

            # 顔のランドマーク描画
            draw_face(results, background)
            # 手のランドマーク描画
            draw_hands(results2, background)

            # FPS の取得と描画
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

            # 画像表示
            cv2.imshow("MediaPipe FaceMesh on White Background", background)

            # "q" で終了
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
```

上記の流れをもう少し噛み砕くと、以下のようになります。

1.

CvFpsCalc

を生成し、FPS 計測用のバッファを用意する。  
 2.

with mp_face_mesh.FaceMesh(...):

で顔検出機能を初期化し、処理が終わると自動的にリソース解放。  
 3.

cap.read()

でカメラ映像を 1 フレーム取得。失敗ならスキップ。  
 4. BGR 形式の画像を RGB に変換して MediaPipe で顔・手を解析。  
 5. 認識結果（顔/手の特徴点）を新規の背景画像 (

background

) に描画。  
 6. FPS を計算して、テキストとして背景画像上に書き込み。  
 7. 最終的に

cv2.imshow()

でウィンドウを表示。  
 8. キー入力を確認し、"q" キーでループを抜ける。

7. ### 最後の後処理
   ```python
   main()
   cap.release()
   cv2.destroyAllWindows()
   ```
   -

main()

を呼び出し、終了後にカメラを解放して全ウィンドウを破棄します。

---

## 改修する際のポイント

1. **検出精度を変更したいとき**

Hands

や

FaceMesh

の

min_detection_confidence

や

min_tracking_confidence

の値を変更してみてください。

2. ## **描画スタイルを変えたいとき**

draw_face

や

draw_hands

内の

mp_drawing.draw_landmarks()

の引数や色などを変えてみましょう。

- たとえば枠線やランドマークの色は

mp_drawing.DrawingSpec

で設定可能です。

3. **背景を変えたいとき**
   - 白色ではなく黒色 (

np.zeros(...)

) にする、あるいは別の画像を読み込んで描画してみてください。

4. ## **FPS 計測のバッファ長を変えたいとき**

cvFpsCalc = CvFpsCalc(buffer_len=10)

の `10` を別の数値に変えると、FPS の計算に使うフレーム数が変わり、表示が安定しやすくなります。

5. **別の機能を付けたいとき**
   - 例: 顔検出と手検出の位置を記録して、特定の動きがあれば何か処理する…などを `draw_〜` 関数の中や

main

の中で行えます。

---

## まとめ

-

app.py

は、顔と手をリアルタイムで認識して描画し、FPS を表示するシンプルなサンプルです。

- 各機能は関数ごとに分割(

draw_face

,

draw_hands

,

main

)されているので、小分割で理解して修正すると効率が良いでしょう。

- FPS 計測には

cvfpscalc.py

の

CvFpsCalc

を使っているので、この仕組みを参考にして別の計測機能の追加もできます。

これらを踏まえてコードを読み進めれば、初心者の方でも比較的改修しやすくなるはずです。もしエラーが出た場合や動作を変更したい場合は、上記のポイントを目安に手を加えてみてください。
