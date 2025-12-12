# Bullettime algorithm

```mermaid
flowchart TD
    input[多視点全方位画像] -->|全方位画像| gen_ppis[透視投影画像群の<br>生成]
    gen_ppis -->|透視投影画像群| pd1{骨格検出}
    pd1 -->|有| get_gaze_point[注視点を取得]
    pd1 -->|無| e1[終了]
    get_gaze_point -->|注視点座標| grouping[注視点をグルーピング]
    grouping -->|グルーピングした注視点の中心座標| gen_gaze_ppi[透視投影画像の生成]
    gen_gaze_ppi -->|透視投影画像| pd2{骨格検出}
    pd2 -->|有| cropping[局所画像の生成]
    pd2 -->|無| e1[終了]
    cropping -->|局所画像| reid[Re Identification] 
    reid --> 3d_reconstraction[3次元復元]
    3d_reconstraction --> bullettime[バレットタイム映像]