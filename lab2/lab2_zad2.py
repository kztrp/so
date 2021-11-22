import multiprocessing as mp
import time
from pathlib import Path
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

LENNA_PATH = Path.cwd() / "lenna.png"

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

def convolve(image, strides: int = 1) -> np.array:
    # Cross Correlation
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    padding = 0
    kernel = np.flipud(np.fliplr(kernel))
    xOutput = int(((image.shape[0] - kernel.shape[0] + 2 * padding) / strides) + 1)
    y_output = int(((image.shape[1] - kernel.shape[1] + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, y_output))

    # Apply Equal Padding to All Sides
    image_padded = np.zeros(
        (image.shape[0] + padding * 2, image.shape[1] + padding * 2)
    )
    image_padded[
    int(padding): int(-1 * padding), int(padding): int(-1 * padding)
    ] = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - kernel.shape[1]:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - kernel.shape[0]:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (
                                kernel
                                * image_padded[
                                  x: x + kernel.shape[0], y: y + kernel.shape[1]
                                  ]
                        ).sum()
                except:
                    break

    plt.imsave("./result.png", output, cmap="binary")


# TEST ZAD 1
if __name__ == '__main__':
    image = plt.imread(str(LENNA_PATH))
    t1 = time.time()
    with mp.Pool(processes=6) as pool:
        pool.apply(convolve, [image,])
    t2 = time.time()

    print(f"Zad 2:  {t2 - t1} seconds")
