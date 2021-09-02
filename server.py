from typing import Sequence, Tuple

import click
import serial
import numpy as np
import cv2
from tqdm import tqdm

OFFSETS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
ORDERS = [[0, 3, 2, 1], [0, 1, 2, 3], [0, 1, 2, 3], [2, 1, 0, 3]]
MAX_DEPTH = 8


def read_gray(img_path: str, size=1024):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)
    return gray


def draw_contour(contour, size):
    canvas = np.ones((size, size)) * 255
    cv2.drawContours(canvas, [contour.astype("int32")], -1, 0, 1)
    cv2.imshow('contour',canvas)


def contour_to_canvas(contour, size):
    return contour * (20 / size) + 6


def get_contour(gray: np.ndarray) -> Sequence[Tuple[float, float]]:
    ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('bin',th)
    contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return np.concatenate(contours[1:]).squeeze()


def hilbert_recursive(point, size, image, depth = 0, offsets = OFFSETS):
    points = [(point[0] + size / 4 * o0, point[1] + size / 4 * o1) for o0, o1 in offsets]
    image_slice = image[
        round(point[1] - size / 2):round(point[1] + size / 2),
        round(point[0] - size / 2):round(point[0] + size / 2),
    ]
    darkness = (255 - np.min(image_slice)) / 255
    if depth / MAX_DEPTH < np.sqrt(darkness):
        return [h for i, p in enumerate(points) for h in hilbert_recursive(
            p, size / 2, image, depth + 1, [offsets[j] for j in ORDERS[i]]
        )]
    return points


def get_hilbert_curve(gray: np.ndarray):
    size = gray.shape[0]
    points = np.array(hilbert_recursive((size/2, size/2), size, gray))
    return points.reshape((-1, 2))


def get_am_line(gray: np.ndarray, lines = 40, samples=500):
    size = gray.shape[0]
    line_width = size // lines 
    step = size // samples
    contour = []
    for i in range(lines):
        for j in range(samples):
            x = int(step * (j + 0.5) + (i % 2) * (size - step * j * 2 -1))
            y = int((i + 0.5) * line_width)
            img_slice = gray[y - line_width//2: y + line_width//2, x - step//2 : x + step//2]
            y += (1 - np.mean(img_slice) / 255) * line_width *0.9  * (1 - 2 * (j % 2))
            contour.append((x, y))
    return np.array(contour)


def evaluate(contour, image):
    contour = contour.reshape(-1, 2)
    canvas = np.ones(image.shape) * 255
    cv2.drawContours(canvas, [contour.astype("int32")], -1, 0, 1)
    canvas = cv2.blur(canvas, (10, 10))
    return np.mean(np.abs(canvas - image))


def get_crow_curve(gray: np.ndarray, points=400, N=2000):
    size = gray.shape[0]
    contour = np.random.randint(0, size, size=(points, 2))
    error = evaluate(contour, gray)
    magnitude = size/3
    progressbar = tqdm(range(N))
    for i in progressbar:
        mask = np.zeros((points, 1))
        mask[np.random.randint(0, points)] = 1
        diffs = np.random.randint(-int(magnitude), int(magnitude), (10, points, 2))
        for diff in diffs:
            new_contour = contour + mask * diff
            new_contour = np.clip(new_contour, 1, size -1)
            new_error = evaluate(new_contour, gray)
            if new_error < error:
                contour = new_contour
                error = new_error
                progressbar.set_description(f"{new_error=}")
    return contour


@click.command()
@click.argument("image", type=click.Path(exists=True))
@click.option("--image-size", default=1024)
@click.option("--serial-port", default="/dev/ttyACM0", type=click.Path(exists=True))
def cli(image: str, image_size: int, serial_port: str):
    gray = read_gray(image, image_size)
    curve = get_contour(gray)
    draw_contour(curve, image_size)
    curve = contour_to_canvas(curve, image_size)
    with serial.Serial(port=serial_port, baudrate=9600) as arduino:
        def command_and_await(x: float, y: float) -> str:
                    arduino.write(f"{x:.2f},{y:.2f};".encode())
                    return arduino.readline().decode('ascii').strip()
        cv2.waitKey(2000)

        click.echo(command_and_await(*curve[0]))
        click.confirm("Are you ready to put pen to paper?", abort=True, default=True)
        progressbar = tqdm(curve) 
        for x, y in progressbar:
            progressbar.set_description(command_and_await(x, y))