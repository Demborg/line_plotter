from math import sin, cos, pi
from typing import Sequence, Tuple

import serial
import numpy as np
import cv2

def get_circle(radius: float) -> Tuple[float, float]:
    return (
        (16 + sin(angle) * radius, 15 - cos(angle) * radius)
        for angle in np.linspace(0, 2*pi, 100)
    )

def main():
    with serial.Serial(port="/dev/ttyACM0", baudrate=9600, timeout=1) as arduino:
        for x, y in get_circle(10):
            print(f"going to: ({x:.2f}, {y:.2f})")
            arduino.write(f"{x:.2f},{y:.2f};".encode())

            print(f"received: {arduino.readline().decode('ascii')}")

        print("Closing port")

if __name__ == "__main__":
    main()