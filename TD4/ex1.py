from PIL import Image
import numpy as np
import math

im = Image.open("maths.png")
im.show()

def euclidean_distance(ts_a, ts_b):
    dist = 0
    for i in range(len(ts_a)):
        # print(f'Comparing index {i}: {ts_a[i]} vs {ts_b[i]}')
        dist += (ts_a[i] - ts_b[i]) ** 2
    return math.sqrt(dist)

def average(points):
  return sum(points) / len(points)

def assign(p, means):
    dist = None
    for mean in means:
        new_dist = euclidean_distance(p, mean)
        dist = new_dist if not dist else dist
        # print(new_dist)
        if dist > new_dist:
            dist = new_dist
    return dist


rgb = np.array(im.convert("RGB").getdata())

print("DISTANCE: ", euclidean_distance(rgb[0], rgb[1]))

print("MOYENNE: ", average([rgb[0],rgb[1],rgb[2]]))

k = 4
means = [np.random.randint(0, 255, 3) for _ in range(k)]

print(assign(rgb[0], means))