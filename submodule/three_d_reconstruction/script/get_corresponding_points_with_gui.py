import sys
import json
import img_utils.img_utils as iu 

img_dir_path = sys.argv[1]
window_scale = float(sys.argv[2])
output_file_path = sys.argv[3]
output_img_dir = sys.argv[4]

imgs = iu.load_imgs(img_dir_path)

save_data = []
drawn_imgs = []
for img in imgs:
    corr_points, drawn_img = iu.get_img_points_with_gui(img, window_scale)
    save_data.append(corr_points.tolist())
    drawn_imgs.append(drawn_img)
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, indent=4)

iu.save_imgs(drawn_imgs, output_img_dir, file_name_pattern=f"camera{{}}_corr_points")
