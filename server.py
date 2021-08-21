from math import sin, cos, pi
from typing import Sequence, Tuple
from pathlib import Path

import serial
import numpy as np
import cv2

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

def main():
    with serial.Serial(port="/dev/ttyACM0", baudrate=9600) as arduino:
        for x, y in get_contour("cat.jpg"):
            print(f"going to: ({x:.2f}, {y:.2f})")
            arduino.write(f"{x:.2f},{y:.2f};".encode())

            print(f"received: {arduino.readline().decode('ascii')}")

        print("Closing port")

if __name__ == "__main__":
    main()