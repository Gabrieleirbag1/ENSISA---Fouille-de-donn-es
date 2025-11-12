from PIL import Image
import numpy as np

im = Image.open("mathys.png")
im.show()

rgb = np.array(im.convert("RGB").getdata())

def K_Means(data, k, maxiters=10000):

    np.random.seed(0)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for _ in range(maxiters):

        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels

k = 5
centroids, labels = K_Means(rgb, k)
segmented_img = centroids[labels].astype(np.uint8)
segmented_img = segmented_img.reshape(im.size[1], im.size[0], 3)
segmented_image = Image.fromarray(segmented_img)
segmented_image.show()