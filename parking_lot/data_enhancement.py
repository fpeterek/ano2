import glob
import os
import random

import click
import cv2 as cv


def convert_img(path: str, outdir: str, vis: bool):
    split = os.path.split(path)
    folder = os.path.split(split[0])[1]
    filecopy = os.path.join(outdir, folder, split[1])

    dark_file = f'{os.path.splitext(split[1])[0]}_dark.png'
    darker_file = f'{os.path.splitext(split[1])[0]}_darker.png'
    light_file = f'{os.path.splitext(split[1])[0]}_light.png'

    dark_file = os.path.join(outdir, folder, dark_file)
    darker_file = os.path.join(outdir, folder, darker_file)
    light_file = os.path.join(outdir, folder, light_file)

    if vis:
        cv.namedWindow('dark', 0)
        cv.namedWindow('darker', 0)
        cv.namedWindow('light', 0)
        cv.namedWindow('orig', 0)

    img = cv.imread(path, 0)

    alpha = random.randint(20, 50) / 100
    beta = random.randint(0, 20)
    dark = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    alpha = random.randint(13, 15) / 100
    beta = random.randint(15, 30)
    darker = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    alpha = random.randint(123, 125) / 100
    beta = random.randint(20, 50)
    light = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    if vis:
        cv.imshow('dark', dark)
        cv.imshow('darker', darker)
        cv.imshow('light', light)
        cv.imshow('orig', img)
        cv.waitKey(0)

    cv.imwrite(filecopy, img)
    cv.imwrite(dark_file, dark)
    cv.imwrite(darker_file, darker)
    cv.imwrite(light_file, light)


@click.command()
@click.option('--free', help='Free parking spaces')
@click.option('--occupied', help='Occupied parking spaces')
@click.option('--dest', help='Destination folder')
@click.option('--vis', type=bool, help='Visualize images', default=False)
def night_images(free, occupied, dest, vis):
    imgs = glob.glob(f'{free}/*')
    imgs += glob.glob(f'{occupied}/*')

    for img in imgs:
        convert_img(img, dest, vis)


@click.group()
def main():
    pass


main.add_command(night_images)


if __name__ == '__main__':
    main()
