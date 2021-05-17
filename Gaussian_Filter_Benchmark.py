import time
import cv2
from numpy.lib import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print('Found 3 channels: {}'.format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print('Converted to gray channels. Size: {}'.format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

        print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()

    return kernel_2D


def gaussian_blur(image, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    return convolution(image, kernel, average=True, verbose=verbose)


def average(lst):
    return sum(lst) / len(lst)


if __name__ == '__main__':

    output_file = open('OUTPUT.TXT', 'a')
    average_time_100kb = []
    average_time_50kb = []
    nr_runs = list(range(1, 51))

    test_number = 1
    while test_number != 51:
        with open('INPUT.txt', 'r') as input_reader:
            output_file.write('Test nr: {}\n'.format(test_number))

            for path in input_reader.readlines():
                path = path.replace('\n', '')

                start_time = time.time()
                image = cv2.imread(path)
                image_to_save = gaussian_blur(image, 5)
                end_time = time.time()

                if path == 'resources\\image_50_kb.jpg':
                    average_time_50kb.append(end_time - start_time)
                    cv2.imwrite('output_images\image_50kb_{}.jpg'. format(test_number), image_to_save)
                else:
                    average_time_100kb.append(end_time - start_time)
                    cv2.imwrite('output_images\image_100kb_{}.jpg'.format(test_number), image_to_save)

                output_file.write('Image name: {}\n'.format(path.replace('resources\\', '')))
                output_file.write(str(end_time - start_time))
                output_file.write('\n\n')

        test_number += 1

    output_file.write('average time of gaussian filtering on 100 kb image (based on set of 50 results) is: {}\n'
                      .format(average(average_time_100kb)))

    output_file.write('average time of gaussian filtering on 50 kb image (based on set of 50 results) is: {}\n'
                      .format(average(average_time_50kb)))

    # create plot
    plt.plot(nr_runs, average_time_50kb, label='50Kb image')
    plt.plot(nr_runs, average_time_100kb, label='100Kb image')
    plt.xlabel('number of runs')
    plt.ylabel('time of execution')

    plt.title('Gaussian Filter Execution Times')
    plt.legend()
    plt.show()

    output_file.close()
