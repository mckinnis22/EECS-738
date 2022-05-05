"""
Requirements:
    - pillow (pip install Pillow)

Example usage:
    - python3 frames2gif.py frames/ output.gif
    - python3 frames2gif.py -h (to see other options)

Notes:
    - you can set fps and looping behavior

...where frames/ is a folder containing any number of image files (e.g. frame_0.png, frame_1.png, ...)
"""

import os.path as osp, os
import re
import argparse
from PIL.Image import Image, open as open_image


NUM_PTN = re.compile(r'[0-9]+')

def make_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument('dir_path', help = 'path to folder containing images')
    argparser.add_argument('output_path', help = 'where to save file')
    argparser.add_argument('--frames_per_second', '-fps', type = float, default = 30, help = 'set fps for final gif')
    argparser.add_argument('--noloop', '-nl', action = 'store_const', const = 1, help = 'make output gif not loop')
    return argparser


def sort(frame_path: str) -> int:
    # NOTE can change this to modify sort behavior
    return int(re.search(NUM_PTN, frame_path).group())


def main():
    argparser = make_argparser()
    args = argparser.parse_args()

    # get args
    dir_path: str = args.dir_path
    output_path: str = args.output_path if args.output_path.endswith('.gif') else args.output_path + '.gif'
    frames_per_second: float = args.frames_per_second
    num_loop: int = int(bool(args.noloop))

    # read in files and convert
    frame_names: list[str] = sorted(os.listdir(dir_path), key = sort)
    frames: list[Image] = list(map(lambda f_name: open_image(osp.join(dir_path, f_name)), frame_names))

    frames[0].save(output_path, format='GIF', append_images=frames[1:], save_all=True, duration=1000/frames_per_second, loop=num_loop)


if __name__ == '__main__':
    main()