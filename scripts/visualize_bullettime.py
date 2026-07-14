import os
import argparse
import json
import numpy as np
import matplotlib
# GUI環境がない場合に備えて、matplotlibのバックエンド設定は必要に応じて行う
# 例: matplotlib.use('Agg') # 保存のみの場合

def load_json(filepath):
    if not filepath or not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_visualizations(pose_data, gaze_data=None, recon_data=None, output_path=None, no_show=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("エラー: 3D描画には matplotlib が必要です。")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    camera_centers = {}
    
    # 1. カメラ姿勢の描画
    if pose_data:
        print("カメラ姿勢を描画中...")
        axis_length = 0.35
        for name, pose in sorted(pose_data.items()):
            R = np.asarray(pose["R"], dtype=float)
            t = np.asarray(pose["t"], dtype=float)

            # カメラ中心 C = -R^T t
            center = -np.dot(R.T, t)
            camera_centers[name] = center

            # カメラ軸方向 (R^T の列ベクトル)
            x_axis = R.T[:, 0]
            y_axis = R.T[:, 1]
            z_axis = R.T[:, 2]

            # カメラ位置プロット
            ax.scatter(center[0], center[1], center[2], s=40, color='tab:blue', edgecolors='k', zorder=5)
            ax.text(center[0], center[1], center[2], f" {name}", fontsize=8, color='black', zorder=10)

            # カメラのローカル座標軸の矢印描画
            ax.quiver(center[0], center[1], center[2], x_axis[0], x_axis[1], x_axis[2], length=axis_length, color='tab:red', normalize=True, alpha=0.7)
            ax.quiver(center[0], center[1], center[2], y_axis[0], y_axis[1], y_axis[2], length=axis_length, color='tab:green', normalize=True, alpha=0.7)
            ax.quiver(center[0], center[1], center[2], z_axis[0], z_axis[1], z_axis[2], length=axis_length, color='tab:blue', normalize=True, alpha=0.7)

    # 2. 代表注視点の描画
    if gaze_data:
        print("代表注視点を描画中...")
        gaze_points = np.asarray(gaze_data, dtype=float)
        if gaze_points.ndim == 2 and gaze_points.shape[1] == 3:
            ax.scatter(gaze_points[:, 0], gaze_points[:, 1], gaze_points[:, 2], 
                       s=100, color='orange', marker='*', edgecolors='black', label='Rep. Gaze Point', zorder=6)
        elif gaze_points.ndim == 1 and gaze_points.shape[0] == 3:
            ax.scatter(gaze_points[0], gaze_points[1], gaze_points[2], 
                       s=100, color='orange', marker='*', edgecolors='black', label='Rep. Gaze Point', zorder=6)

    # 3. 3D復元結果の描画
    if recon_data:
        print("3D復元データを描画中...")
        # 3D復元データは { "0": { "gaze_point_3d": [...], "length_points_3d": [...], "cameras": [...] }, ... } のような辞書
        for key, item in recon_data.items():
            gaze_3d = np.asarray(item.get("gaze_point_3d"), dtype=float)
            length_pts = np.asarray(item.get("length_points_3d", []), dtype=float)
            recon_cameras = item.get("cameras", [])

            # 3D注視点プロット
            if gaze_3d.shape == (3,):
                ax.scatter(gaze_3d[0], gaze_3d[1], gaze_3d[2], s=60, color='magenta', marker='o', edgecolors='k', label=f'Gaze Point {key}' if key == '0' else "", zorder=7)
                # 注視点の真上にクラスタ番号（キー）を描画（Z座標を少し上にオフセット）
                ax.text(gaze_3d[0], gaze_3d[1], gaze_3d[2] + 0.12, f"C{key}", fontsize=9, color='magenta', fontweight='bold', ha='center', va='bottom', zorder=10)

            # 長さ計測の3D点群 (折れ線)
            if length_pts.ndim == 2 and length_pts.shape[1] == 3:
                ax.plot(length_pts[:, 0], length_pts[:, 1], length_pts[:, 2], color='darkcyan', linewidth=2, label='Length Trajectory' if key == '0' else "", zorder=4)
                ax.scatter(length_pts[:, 0], length_pts[:, 1], length_pts[:, 2], s=10, color='darkcyan', zorder=4)



    # 4. 描画範囲とアスペクト比の設定
    all_points = []
    if camera_centers:
        all_points.extend(camera_centers.values())
    if gaze_data:
        gaze_points = np.asarray(gaze_data, dtype=float)
        if gaze_points.ndim == 2:
            all_points.extend(gaze_points)
        elif gaze_points.ndim == 1:
            all_points.append(gaze_points)
    if recon_data:
        for item in recon_data.values():
            gaze_3d = np.asarray(item.get("gaze_point_3d"), dtype=float)
            if gaze_3d.shape == (3,):
                all_points.append(gaze_3d)
            length_pts = np.asarray(item.get("length_points_3d", []), dtype=float)
            if length_pts.ndim == 2 and length_pts.shape[1] == 3:
                all_points.extend(length_pts)

    if all_points:
        all_points = np.asarray(all_points)
        center = all_points.mean(axis=0)
        max_range = np.ptp(all_points, axis=0).max()
        if max_range == 0:
            max_range = 1.0
        half_range = max_range / 2.0 + 0.5
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)

        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Bullettime Reconstruction 3D Plot")
    
    # 凡例の追加
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.tight_layout()

    # 画像の保存
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"結果を画像として保存しました: {output_path}")

    # 表示
    if not no_show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bullettimeの実行結果（カメラ姿勢、代表注視点、3D復元結果）を3Dプロットする")
    parser.add_argument("-p", "--pose-json", required=True, help="aggregated camera pose json file path (e.g. images_aggregated_pose.json)")
    parser.add_argument("-g", "--gaze-json", default=None, help="representative gaze points json file path (e.g. representative_gaze_points.json)")
    parser.add_argument("-r", "--reconstruction-json", default=None, help="3d reconstruction json file path (e.g. reconstruction_3d.json)")
    parser.add_argument("-o", "--output", default=None, help="output path to save plot image")
    parser.add_argument("--no-show", action="store_true", help="do not show interactive matplotlib window")
    args = parser.parse_args()

    # GUIがない環境（ヘッドレス環境）で no-show が指定された場合、matplotlib バックエンドを Agg に切り替える
    if args.no_show or not os.environ.get("DISPLAY", None):
        if not os.name == 'nt':  # WindowsはDISPLAY環境変数が無いため無視
            matplotlib.use('Agg')
    if args.no_show and os.name == 'nt':
        # Windows環境でもGUIを非表示にするためにAggを適用
        matplotlib.use('Agg')

    pose_data = load_json(args.pose_json)
    gaze_data = load_json(args.gaze_json)
    recon_data = load_json(args.reconstruction_json)

    if not pose_data:
        print("エラー: 有効なカメラ姿勢データ (pose-json) が必要です。")
        exit(1)

    plot_visualizations(
        pose_data=pose_data,
        gaze_data=gaze_data,
        recon_data=recon_data,
        output_path=args.output,
        no_show=args.no_show
    )
# 正しいTclのパスを環境変数に設定
# $env:TCL_LIBRARY="C:\Users\naoki\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\tcl\tcl8.6"
# スクリプトの実行
# uv run .\scripts\visualize_bullettime.py -p c:\Users\naoki\Prog\input\colmap\2026-06-23\images_aggregated_pose.json