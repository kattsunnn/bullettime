import click
from pathlib import Path
import numpy as np
import img_utils.img_utils as iu
from generate_ppi import generate_ppi_with_gui
from e2p import E2P

@click.command
@iu.prepare_io_path
@click.option('-ws', '--window_scale', required=True, type=float, help='スケール（必須）')
@click.option('-fwd', '--fov-w_deg', type=float, required=True, help='画角（横）')
@click.option('-fhd', '--fov-h_deg', type=float, required=True, help='画角（縦）')
def main(input_path, output_path, window_scale, fov_w_deg, fov_h_deg):
    src_img = iu.load_imgs(input_path)
    ppi, r_mat = generate_ppi_with_gui(src_img, window_scale, fov_w_deg, fov_h_deg)
    # print(E2P.r_mat_to_angle(r_mat))
    iu.show_imgs(ppi)
    file_name = Path(input_path).stem + "_ppi"
    iu.save_imgs(ppi, output_path, file_name=file_name)
    file_name = Path(input_path).stem + "_ppi_r"
    np.savetxt(Path(output_path)/Path(file_name).with_suffix(".dat"), r_mat)

main()