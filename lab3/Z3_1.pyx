from matplotlib import pyplot as plt

import numpy as np
cimport numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

np.import_array()
DTYPE = np.int
ctypedef np.int_t DTYPE_t
def convolve(image, kernel, padding=1, strides=1):
    kernel = np.flipud(np.fliplr(kernel))
    xOutput = int(((image.shape[0] - kernel.shape[0] + 2 * padding)) + 1)
    yOutput = int(((image.shape[1] - kernel.shape[1] + 2 * padding)) + 1)
    cdef output = np.zeros((xOutput, yOutput))
    cdef imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
    imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    workers = 4
    cdef yshapes = np.linspace(0, image.shape[1], workers).astype(int)
    with ProcessPoolExecutor(max_workers=workers-1) as executor:
        tasks = []
        for i in range(workers-1):
            future = executor.submit(partialCon, image, yshapes[i], yshapes[i+1], imagePadded, kernel)
            # future.add_done_callback(done_function)
            tasks.append(future)
        for future in as_completed(tasks):
            output += future.result()
    output = (output <= 0.5) * 1
    plt.imsave('result2.png', output, cmap='binary')
def partialCon(image, ymin, ymax, imagePadded, kernel, padding=1, strides=1):
    xOutput = int(((image.shape[0] - kernel.shape[0] + 2 * padding)) + 1)
    yOutput = int(((image.shape[1] - kernel.shape[1] + 2 * padding)) + 1)
    cdef output = np.zeros((xOutput, yOutput))
    for y in range(ymin, ymax):
        if y > imagePadded.shape[1] - kernel.shape[1]:
            break
        if y % strides == 0:
            for x in range(image.shape[0]):
                if x > imagePadded.shape[0] - kernel.shape[0]:
                    break
                try:
                    if y % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + kernel.shape[0],
                    y: y + kernel.shape[1]]).sum()
                except:
                    break
    return output

def main():
    cdef kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    cdef image = plt.imread('lenna.png')
    t1 = time.time()
    out = convolve(image, kernel)
    t2 = time.time()

    print("Zadanie 2.1 (wersja zrownoleglona)\tCzas:{} s".format(t2-t1))

if __name__=="__main__":
    main()
