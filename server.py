from typing import Sequence, Tuple

import serial
import numpy as np
import cv2

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
    cv2.waitKey(0)


def contour_to_canvas(contour, size):
    return contour * (20 / size) + 6


def get_contour(img_path: str) -> Sequence[Tuple[float, float]]:
    gray = read_gray(img_path)
    ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('bin',th)
    contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contour = np.concatenate(contours[1:]).squeeze()
    import ipdb; ipdb.set_trace()
    draw_contour(contour, gray.shape[0])
    return contour_to_canvas(contour, gray.shape[0])


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


def get_hilbert_curve(img_path: str):
    gray = read_gray(img_path)
    size = gray.shape[0]
    points = np.array(hilbert_recursive((size/2, size/2), size, gray))
    contour = points.reshape((-1, 1, 2))
    draw_contour(contour, size)
    return contour_to_canvas(points, size)


def get_am_line(img_path: str, lines = 40, samples=500):
    gray = read_gray(img_path, 1000)
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
    contour = np.array(contour)
    draw_contour(contour, size)
    return contour_to_canvas(contour, size)


def main():
    stop = True
    with serial.Serial(port="/dev/ttyACM0", baudrate=9600) as arduino:
        for x, y in get_am_line("k.png", lines=40, samples=200):
            print(f"going to: ({x:.2f}, {y:.2f})")
            arduino.write(f"{x:.2f},{y:.2f};".encode())
            print(f"received: {arduino.readline().decode('ascii')}")

            if stop and input("Are you ready [Y,n]") != "n":
                stop = False


if __name__ == "__main__":
    main()