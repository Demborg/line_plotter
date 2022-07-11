import json
from typing import Sequence, Tuple

import click
import serial
import numpy as np
import cv2
from tqdm import tqdm

OFFSETS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
ORDERS = [[0, 3, 2, 1], [0, 1, 2, 3], [0, 1, 2, 3], [2, 1, 0, 3]]

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
    contour *= (22 / size)
    contour[:, 0] += 7
    contour[:, 1] += 16
    return contour


def get_contour(gray: np.ndarray) -> Sequence[Tuple[float, float]]:
    ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('bin',th)
    contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return np.concatenate(contours[1:]).squeeze()


def get_hilbert_curve(gray: np.ndarray, max_depth=8):
    def hilbert_recursive(point, size, depth = 0, offsets = OFFSETS):
        points = [(point[0] + size / 4 * o0, point[1] + size / 4 * o1) for o0, o1 in offsets]
        image_slice = gray[
            round(point[1] - size / 2):round(point[1] + size / 2),
            round(point[0] - size / 2):round(point[0] + size / 2),
        ]
        darkness = (255 - np.min(image_slice)) / 255
        if depth / max_depth < np.sqrt(darkness):
            return [h for i, p in enumerate(points) for h in hilbert_recursive(
                p, size / 2, depth + 1, [offsets[j] for j in ORDERS[i]]
            )]
        return points


    size = gray.shape[0]
    points = np.array(hilbert_recursive((size/2, size/2), size))
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


def get_crow_curve(gray: np.ndarray, points=300, iterations=20000):
    size = gray.shape[0]
    contour = np.random.randint(0, size, size=(points, 2))
    error = evaluate(contour, gray)
    magnitude = size/3
    progressbar = tqdm(range(iterations))
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


def get_circle_curve(gray: np.ndarray, num_nodes=100, points=300, iterations=20000):
    generator = np.random.default_rng(seed=42)
    size = gray.shape[0]
    angles = np.linspace(0, 2 * np.pi, num_nodes)
    nodes = ((np.stack([np.sin(angles), np.cos(angles)]) + 1) * size / 2).T
    contour = generator.choice(nodes, points)
    error = evaluate(contour, gray)
    progressbar = tqdm(range(iterations))
    for _ in progressbar:
        for index in np.random.random_integers(len(contour) - 1, size=(10)):
            new_contour = contour.copy()
            new_contour[index, :] = generator.choice(nodes, 1)
            new_error = evaluate(new_contour, gray)
            if new_error < error:
                contour = new_contour
                error = new_error
                progressbar.set_description(f"{new_error=}")
    return contour


def get_ray_circle_curve(gray: np.ndarray, num_nodes=100, points=300, iterations=20000):
    size = gray.shape[0]
    angles = np.linspace(0, 2 * np.pi, num_nodes)
    nodes = ((np.stack([np.sin(angles), np.cos(angles)]) + 1) * size / 2).T
    votes = np.zeros((num_nodes, num_nodes))
    samples = np.zeros(gray.shape)
    for _ in tqdm(range(iterations)):
        coord = np.random.random_integers(0, size - 1, size=(2))
        if sum((coord - size / 2) ** 2) > (size/2) ** 2: 
            continue
        samples[coord[0], coord[1]] += 1
        directions = np.random.random(255 - gray[coord[0], coord[1]]) * np.pi
        for dir in directions:
            normal = np.array((np.cos(dir), np.sin(dir)))
            x = 2 * coord / size - 1
            thing = x @ normal
            l1 = - thing + np.sqrt(thing ** 2 - x @ x + 1)
            l2 = - thing - np.sqrt(thing ** 2 - x @ x + 1)
            r1 = x + normal * l1
            r2 = x + normal * l2
            i = np.argmin(np.linalg.norm(nodes - (r1 + 1) * size / 2, axis=1))
            j = np.argmin(np.linalg.norm(nodes - (r2 + 1) * size / 2, axis=1))
            if i != j:
                votes[i, j] += 1 / np.linalg.norm(nodes[i, :] - nodes[j, :])
                votes[j, i] += 1 / np.linalg.norm(nodes[i, :] - nodes[j, :])
    cv2.imshow("votes", votes / np.max(votes))
    cv2.imshow("samples", samples / np.max(samples))

    contour = []
    index = 0
    indices = np.array(range(num_nodes))
    for _ in range(points):
        index = np.random.choice(indices, p=votes[index] / sum(votes[index]))
        contour.append(nodes[index])

    return np.array(contour)

def get_awsome_simon_line(gray: np.ndarray):
    toVisit = (gray / 255) ** 0.5 < np.random.random(size=gray.shape)
    visited = np.full(toVisit.shape, False) 
    initial_sum = np.sum(toVisit)
    p = np.array([0, 0])
    contour = []
    bar = tqdm(total=np.sum(toVisit))
    current_sum = np.sum(toVisit)
    while current_sum / initial_sum > 0.001:
        bar.update()
        angles = np.random.random(100) * 2 * np.pi
        directions = np.stack([np.sin(angles), np.cos(angles)]).T
        distances = np.zeros(angles.shape)
        for r in range(1, len(toVisit) * 2):
            distances += 1

            for i, dir in enumerate(directions):
                while True:
                    q = np.array(p + np.round(dir * distances[i]), dtype=int)
                    if not all(0 <= x < len(toVisit) for x in q):
                        break
                    if visited[q[0], q[1]]:
                        distances[i] +=1
                    else:
                        break
                if not all(0 <= x < len(toVisit) for x in q):
                        continue
                if toVisit[q[0], q[1]]:
                    current_sum -= 1
                    contour.append([q[1], q[0]])
                    toVisit[q[0], q[1]] = False
                    visited[q[0], q[1]] = True
                    p = q
                    break
            else:
                continue
            break
    return np.array(contour).astype(np.float64)


methods = {"crow": get_crow_curve, "am": get_am_line, "hilbert": get_hilbert_curve, "contour": get_contour, "circle": get_ray_circle_curve, "simon": get_awsome_simon_line}


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("image", type=click.Path(exists=True))
@click.option("--image-size", default=1024)
@click.option("--method", default="crow", type=click.Choice(methods.keys()))
@click.option("--serial-port", default="/dev/ttyACM0", type=click.Path(exists=True))
@click.option("--method_json", type=str)
def cli(image: str, image_size: int, method: str, serial_port: str, method_json: str):
    gray = read_gray(image, image_size)
    kwargs = json.loads(method_json) if method_json else {}
    curve = methods[method](gray, **kwargs)
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