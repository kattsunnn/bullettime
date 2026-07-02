import argparse
import cv2
import os
from pathlib import Path

def draw_center_point(
    image_path: str,
    output_path: str = None,
    color: tuple = (0, 0, 255),  # BGR形式 (デフォルト: 赤)
    radius: int = 10,
    thickness: int = -1,        # -1 は塗りつぶし
    show: bool = False
):
    """
    指定された画像ファイルの中心に点を描き、保存（およびオプションで表示）する関数
    """
    # 画像の読み込み
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 画像ファイルが見つからないか、読み込めませんでした: {image_path}")
        return

    # 画像サイズから中心座標を計算
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    print(f"Image Size: {w}x{h} | Center Coordinate: ({cx}, {cy})")

    # 中心に点を描画
    cv2.circle(img, (cx, cy), radius, color, thickness)

    # 保存パスの決定
    if output_path is None:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_center{p.suffix}")

    # 保存先ディレクトリの作成
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 画像を書き込み
    cv2.imwrite(output_path, img)
    print(f"中心に点を描画した画像を保存しました: {output_path}")

    # オプションで画像を表示する
    if show:
        cv2.imshow("Check Center Point", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="画像の中心に点を描画するプログラム")
    parser.add_argument("-i", "--input", required=True, help="入力画像のパス")
    parser.add_argument("-o", "--output", default=None, help="出力画像のパス (指定しない場合は入力と同じフォルダに '_center' を付加して保存)")
    parser.add_argument("-r", "--radius", type=int, default=5, help="点の半径 (デフォルト: 5)")
    parser.add_argument("-c", "--color", nargs=3, type=int, default=[0, 0, 255], 
                        help="点の色を BGR 順のスペース区切りで指定。例: 青=[255, 0, 0]、赤=[0, 0, 255] (デフォルト: 0 0 255)")
    parser.add_argument("-t", "--thickness", type=int, default=-1, help="描画する線の太さ。-1で塗りつぶし (デフォルト: -1)")
    parser.add_argument("--show", action="store_true", help="作成した画像をウィンドウで表示する")

    args = parser.parse_args()

    # 引数の色をタプルに変換
    color_tuple = tuple(args.color)

    draw_center_point(
        image_path=args.input,
        output_path=args.output,
        color=color_tuple,
        radius=args.radius,
        thickness=args.thickness,
        show=args.show
    )

if __name__ == "__main__":
    main()
