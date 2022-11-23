import glob
import os
import random
import math

import click
import cv2 as cv
import scipy as sp


def rotate_result(fn):
    def with_rotation(img):
        img = fn(img)
        avg = 0
        for x in range(80):
            for y in range(80):
                avg += img[y][x]

        avg /= 80*80

        angle = random.randint(-40, 40)
        return sp.ndimage.rotate(img, angle, reshape=False, cval=avg)

    return with_rotation


@rotate_result
def dark_img(img):
    alpha = random.randint(20, 50) / 100
    beta = random.randint(0, 20)
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)


@rotate_result
def darker_img(img):
    alpha = random.randint(13, 15) / 100
    beta = random.randint(15, 30)
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)


@rotate_result
def light_img(img):
    alpha = random.randint(110, 120) / 100
    beta = random.randint(20, 50)
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)


def lin_shade(light, shade):
    shade_len = random.randint(30, 45)

    ranges = [
            (range(shade_len), range(80)),
            (range(80-shade_len, 80), range(80)),
            (range(80), range(shade_len)),
            (range(80), range(80-shade_len, 80)),
            ]

    x_range, y_range = random.choice(ranges)

    for x in x_range:
        for y in y_range:
            light[y][x] = shade[y][x]

    return light


def horizonal_shade(x, y):
    return y, x


def vertical_shade(x, y):
    return x, y


def wave_shade(light, shade):
    width = random.randint(2, 10) * math.pi
    px_step = width / 80
    wavefn = random.choice((math.sin, math.cos))
    dirfn = random.choice((horizonal_shade, vertical_shade))

    for x in range(80):
        phase = px_step * x
        light_ratio = (1 + wavefn(phase)) / 2
        dark_ratio = 1 - light_ratio
        for y in range(80):
            tx, ty = dirfn(x, y)
            light_px = light[tx][ty] * light_ratio
            dark_px = shade[tx][ty] * dark_ratio
            light[tx][ty] = light_px + dark_px

    return light


def shade_img(img):
    light = cv.convertScaleAbs(img, alpha=1.2, beta=25)
    shade = cv.convertScaleAbs(img, alpha=0.6, beta=25)

    fn = random.choice((lin_shade, wave_shade))

    return fn(light, shade)


@rotate_result
def noise_img(img):
    center_coordinates = (random.randint(25, 80-25), random.randint(25, 80-25))
    axesLength = (random.randint(7, 13), random.randint(7, 13))
    angle = random.randint(0, 180)
    startAngle = 0
    endAngle = 360
    color = random.choice(((210, 210, 210), (30, 30, 30)))
    thickness = -1

    copy = cv.convertScaleAbs(img, alpha=1.0, beta=25)

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
