import glob
import os
import random

import click
import cv2 as cv


def dark_img(img):
    alpha = random.randint(20, 50) / 100
    beta = random.randint(0, 20)
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)


def darker_img(img):
    alpha = random.randint(13, 15) / 100
    beta = random.randint(15, 30)
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)


def light_img(img):
    alpha = random.randint(123, 125) / 100
    beta = random.randint(20, 50)
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)


def shade_img(img):

    shade_len = random.randint(30, 45)

    ranges = [
            (range(shade_len), range(80)),
            (range(80-shade_len, 80), range(80)),
            (range(80), range(shade_len)),
            (range(80), range(80-shade_len, 80)),
            ]

    x_range, y_range = random.choice(ranges)

    light = cv.convertScaleAbs(img, alpha=1.2, beta=25)
    shade = cv.convertScaleAbs(img, alpha=0.6, beta=25)

    for x in x_range:
        for y in y_range:
            light[y][x] = shade[y][x]

    return light


def noise_img(img):
    center_coordinates = (random.randint(25, 80-25), random.randint(25, 80-25))
    axesLength = (random.randint(7, 16), random.randint(7, 16))
    angle = random.randint(0, 180)
    startAngle = 0
    endAngle = 360
    color = (210, 210, 210)
    thickness = -1

    copy = img.copy()

    return cv.ellipse(copy, center_coordinates, axesLength, angle,
                      startAngle, endAngle, color, thickness)


def convert_img(path: str, outdir: str, vis: bool):
    split = os.path.split(path)
    folder = os.path.split(split[0])[1]
    filecopy = os.path.join(outdir, folder, split[1])

    dark_file = f'{os.path.splitext(split[1])[0]}_dark.png'
    darker_file = f'{os.path.splitext(split[1])[0]}_darker.png'
    light_file = f'{os.path.splitext(split[1])[0]}_light.png'
    shade_file = f'{os.path.splitext(split[1])[0]}_shade.png'
    noise_file = f'{os.path.splitext(split[1])[0]}_noise.png'

    dark_file = os.path.join(outdir, folder, dark_file)
    darker_file = os.path.join(outdir, folder, darker_file)
    light_file = os.path.join(outdir, folder, light_file)
    shade_file = os.path.join(outdir, folder, shade_file)
    noise_file = os.path.join(outdir, folder, noise_file)

    if vis:
        cv.namedWindow('dark', 0)
        cv.namedWindow('darker', 0)
        cv.namedWindow('light', 0)
        cv.namedWindow('orig', 0)
        cv.namedWindow('shade', 0)
        cv.namedWindow('noise', 0)

    img = cv.imread(path, 0)

    dark = dark_img(img)
    darker = darker_img(img)
    light = light_img(img)
    # TODO: These two make things worse
    shade = shade_img(img)
    noise = noise_img(img)

    if vis:
        cv.imshow('dark', dark)
        cv.imshow('darker', darker)
        cv.imshow('light', light)
        cv.imshow('orig', img)
        cv.imshow('shade', shade)
        cv.imshow('noise', noise)
        cv.waitKey(0)

    cv.imwrite(filecopy, img)
    cv.imwrite(dark_file, dark)
    cv.imwrite(darker_file, darker)
    cv.imwrite(light_file, light)
    cv.imwrite(shade_file, shade)
    cv.imwrite(noise_file, noise)


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
