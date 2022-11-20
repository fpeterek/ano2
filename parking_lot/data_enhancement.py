import glob
import os
import random

import click
import cv2 as cv


def convert_img(path: str, outdir: str):
    split = os.path.split(path)
    folder = os.path.split(split[0])[1]
    filecopy = os.path.join(outdir, folder, split[1])
    outfile = f'{os.path.splitext(split[1])[0]}_dark.png'

    outfile = os.path.join(outdir, folder, outfile)

    cv.namedWindow('Zpiceny eduroam', 0)
    cv.namedWindow('orig', 0)

    img = cv.imread(path, 0)

    alpha = random.randint(15, 50) / 100
    beta = random.randint(5, 11)
    dark = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    # print(filecopy)
    # print(outfile)

    cv.imshow('Zpiceny eduroam', dark)
    cv.imshow('orig', img)
    cv.waitKey(0)

    cv.imwrite(filecopy, img)
    cv.imwrite(outfile, dark)


@click.command()
@click.option('--free', help='Free parking spaces')
@click.option('--occupied', help='Occupied parking spaces')
@click.option('--dest', help='Destination folder')
def night_images(free, occupied, dest):
    imgs = glob.glob(f'{free}/*')
    imgs += glob.glob(f'{occupied}/*')

    for img in imgs:
        convert_img(img, dest)


@click.group()
def main():
    pass


main.add_command(night_images)


if __name__ == '__main__':
    main()
