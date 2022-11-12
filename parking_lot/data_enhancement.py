import glob
import os

import click
import cv2 as cv


def convert_img(path: str):
    outfile = f'{os.path.splitext(path)[0]}_dark.png'

    cv.namedWindow('Zpiceny eduroam', 0)

    img = cv.imread(path, 0)
    # TODO: Darken image, save to file

    cv.imshow('Zpiceny eduroam', img)
    cv.waitKey(0)


@click.command()
@click.option('--free', help='Free parking spaces')
@click.option('--occupied', help='Occupied parking spaces')
def night_images(free, occupied):
    imgs = glob.glob(f'{free}/*')
    imgs += glob.glob(f'{occupied}/*')

    for img in imgs:
        convert_img(img)


@click.group()
def main():
    pass


main.add_command(night_images)
