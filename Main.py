import matplotlib.pyplot as plt
import os
from skimage import morphology
from skimage import data
from skimage.color import label2rgb
from skimage.color import rgb2grey
from scipy import ndimage as ndi
from skimage.filters import sobel
import numpy as np

IMAGE_DIR = 'Images/'
OUTPUT_FILENAME = 'result.jpg'


def get_images(image_dir):
    """
    Load images into memory
    :param image_dir: directory with images to transform
    :return: array of raw image data
    """

    images_filenames = os.listdir(image_dir)
    images = []

    for image_filename in images_filenames:
        # Get absolute filenames
        abs_image_filename = os.path.abspath(os.path.join(IMAGE_DIR, image_filename))
        # Read in images from list to images table
        images.append(data.imread(abs_image_filename, as_grey=False))

    # return images table
    return images


def plot_images(images, output_filename, rows=5, columns=5):
    """
    Draw images in a mosaic and save to output filename
    :param columns: number of columns in mosaic (default 5)
    :param rows: number of rows in mosaic (default 5)
    :param images: array of images to contain in a mosaic
    :param output_filename: name of the file to save mosaic into
    """

    fig = plt.figure(num=None, figsize=(30, 30), dpi=80)

    for index, image in enumerate(images):
        a = fig.add_subplot(rows, columns, index + 1)

        plt.imshow(image, cmap=plt.cm.gray)
        a.set_title(str(index))
        plt.axis("off")

    fig.savefig(output_filename)


def transform_image(image):
    """
    :param image: original image
    :return: transformed image
    """

    transformable_image = rgb2grey(image)  # transformation beginning

    # median_transform = median(before_transform, disk(10))
    elevation_map = sobel(transformable_image)
    markers = np.ones_like(transformable_image)
    markers[transformable_image < 0.4] = 2
    markers[transformable_image > 0.7] = 0

    transformable_image = morphology.watershed(elevation_map, markers)

    segmentation = ndi.binary_fill_holes(transformable_image -1)
    #labeled, _ = ndi.label(segmentation)

    #image_overlay = label2rgb(labeled, image=image)

    return segmentation


def transform(images):
    """
    :param images: array of images
    :return: transformed array of images
    """

    for index, image in enumerate(images):
        images[index] = transform_image(image)
    return images


def main():
    plot_images(transform(get_images(IMAGE_DIR)), OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
