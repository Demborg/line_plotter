from math import sin, cos, pi
from typing import Sequence, Tuple

import serial
import numpy as np
import cv2

OFFSETS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
ORDERS = [[0, 3, 2, 1], [0, 1, 2, 3], [0, 1, 2, 3], [2, 1, 0, 3]]
MAX_DEPTH = 7

def get_circle(radius: float) -> Sequence[Tuple[float, float]]:
    return (
        (16 + sin(angle) * radius, 15 - cos(angle) * radius)
        for angle in np.linspace(0, 2*pi, 100)
    )

def get_contour(img_path: str) -> Sequence[Tuple[float, float]]:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('bin',th)
    contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    edges = np.zeros(gray.shape)
    cv2.drawContours(edges, contours[1:], -1, 255, 3)
    cv2.imshow('contour',edges)
    contours = np.concatenate(contours[1:]).astype('float32').squeeze()
    contours[:, 0] /= img.shape[0]
    contours[:, 1] /= img.shape[1]
    contours *= 20
    contours += 6
    cv2.waitKey(0)
    return contours

def hilbert_recursive(point, size, image, depth = 0, offsets = OFFSETS):
    points = [(point[0] + size * o0, point[1] + size * o1) for o0, o1 in offsets]
    image_slice = image[
        round(point[1] - size * 2):round(point[1] + size * 2),
        round(point[0] - size * 2):round(point[0] + size * 2),
    ]
    darkness = (255 - np.min(image_slice)) / 255
    if depth / MAX_DEPTH < np.sqrt(darkness):
        return [h for i, p in enumerate(points) for h in hilbert_recursive(
            p, size/2, image, depth + 1, [offsets[j] for j in ORDERS[i]]
        )]
    return points

def get_hilbert_curve(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1024, 1024))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    size = gray.shape[0] / 2
    points = np.array(hilbert_recursive((size, size), size/2, gray))
    contours = [points.reshape((-1, 1, 2)).astype("int32")]
    canvas = np.zeros(gray.shape)
    cv2.drawContours(canvas, contours, -1, 255, 1)
    cv2.imshow('contour',canvas)

    points /= gray.shape[0]
    points *= 20
    points += 6
    cv2.waitKey(0)

    return points


def main():
    with serial.Serial(port="/dev/ttyACM0", baudrate=9600) as arduino:
        for x, y in get_hilbert_curve("cat.jpg"):
            print(f"going to: ({x:.2f}, {y:.2f})")
            arduino.write(f"{x:.2f},{y:.2f};".encode())

            print(f"received: {arduino.readline().decode('ascii')}")

        print("Closing port")

if __name__ == "__main__":
    main()