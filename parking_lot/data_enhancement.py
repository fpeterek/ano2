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


def maybe_noise(fn):
    def maybe_noise(img):
        if random.random() < 0.3:
            f = random.choice((draw_ellipse, draw_line))
            img = f(img.copy())
        return fn(img)

    return maybe_noise


@rotate_result
@maybe_noise
def dark_img(img):
    alpha = random.randint(20, 50) / 100
    beta = random.randint(0, 20)
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)


@rotate_result
@maybe_noise
def darker_img(img):
    alpha = random.randint(13, 15) / 100
    beta = random.randint(-10, 15)
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)


@rotate_result
@maybe_noise
def light_img(img):
    alpha = random.randint(115, 150) / 100
    beta = random.randint(15, 50)
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)


def coord_id(x):
    return x


def coord_inv(x):
    return 79 - x


def horizonal_shade(x, y):
    return y, x


def vertical_shade(x, y):
    return x, y


def lin_shade(light, shade):
    shade_len = random.randint(25, 45)
    falloff_base = 15

    dirfn = random.choice((horizonal_shade, vertical_shade))
    sidefn = random.choice((coord_id, coord_inv))

    for x in range(shade_len):
        shade_ratio = min(math.log(shade_len - x) / math.log(falloff_base), 1)
        light_ratio = 1 - shade_ratio
        for y in range(80):
            sx = sidefn(x)
            dx, dy = dirfn(sx, y)
            light_px = light[dy][dx] * light_ratio
            shade_px = shade[dy][dx] * shade_ratio

            light[dy][dx] = light_px + shade_px

    return light


def mask_shade(light, shade):
    mask = cv.imread('data/shadow_mask.png', 0)

    for x in range(80):
        for y in range(80):
            if mask[y][x] != 255:
                light[y][x] = shade[y][x]

    return light


def wave_shade(light, shade):
    width = random.randint(2, 8) * math.pi
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


@maybe_noise
def shade_img(img):
    alpha_l = random.randint(10, 12) / 10
    alpha_s = random.randint(30, 55) / 100
    beta = random.randint(0, 25)

    light = cv.convertScaleAbs(img, alpha=alpha_l, beta=beta)
    shade = cv.convertScaleAbs(img, alpha=alpha_s, beta=beta)

    fn = random.choice((lin_shade, wave_shade, mask_shade))

    return fn(light, shade)


def draw_ellipse(img):
    center = (random.randint(25, 80-25), random.randint(25, 80-25))
    axes_len = (random.randint(7, 13), random.randint(7, 13))
    angle = random.randint(0, 180)
    color = random.choice((220, 35))
    inv_color = 255 - color
    color = (color, color, color)
    inv_color = (inv_color, inv_color, inv_color)

    axes_len2 = (axes_len[0] + 2, axes_len[1] + 2)

    cv.ellipse(img, center, axes_len2, angle, 0, 360, inv_color, -1)

    return cv.ellipse(img, center, axes_len, angle, 0, 360, color, -1)


def rand_coord():
    return random.randint(20, 60)


def rand_left():
    return 0, rand_coord()


def rand_right():
    return 80, rand_coord()


def rand_up():
    y, x = rand_left()
    return x, y


def rand_down():
    y, x = rand_right()
    return x, y


def draw_line(img):
    fns = [rand_left, rand_right, rand_up, rand_down]
    start, end = random.sample(fns, 2)
    start, end = start(), end()
    cv.line(img, start, end, (30, 30, 30), 4)
    cv.line(img, start, end, (225, 225, 225), 2)

    return img


@rotate_result
def noise_img(img):
    fn = random.choice((draw_ellipse, draw_line))
    alpha = random.randint(90, 110) / 100
    beta = random.randint(-15, 15)
    copy = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    return fn(copy)


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
def enhance(free, occupied, dest, vis):
    imgs = glob.glob(f'{occupied}/*')
    imgs += glob.glob(f'{free}/*')

    # Shuffle only serves visualization purposes
    if vis:
        random.shuffle(imgs)

    for img in imgs:
        convert_img(img, dest, vis)


@click.group()
def main():
    pass


main.add_command(enhance)


if __name__ == '__main__':
    main()
