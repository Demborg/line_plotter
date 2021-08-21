from math import sin, cos, pi
from typing import Tuple

import serial
import numpy as np

def get_coordinates(angle: float, radius: float) -> Tuple[float, float]:
    return (16 + sin(angle) * radius, 15 - cos(angle) * radius)


def main():
    with serial.Serial(port="/dev/ttyACM0", baudrate=9600, timeout=1) as arduino:
        for angle in np.linspace(0, 2*pi, 100):
            x, y = get_coordinates(
                angle,
                10,
            )
            print(f"going to: ({x:.2f}, {y:.2f})")
            arduino.write(f"{x:.2f},{y:.2f};".encode())

            print(f"received: {arduino.readline().decode('ascii')}")

        print("Closing port")

if __name__ == "__main__":
    main()