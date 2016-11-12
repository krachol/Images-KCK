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

    for image_filename in os.listdir(image_dir):
        # Get absolute filenames
        abs_image_filename = os.path.abspath(os.path.join(IMAGE_DIR, image_filename))
        # Read in images from list to images table
        yield data.imread(abs_image_filename, as_grey=False)


def get(i):
    return i


def get_next_filename(output_filename):
    return output_filename


def plot_images(images, output_filename, rows=6, columns=4):
    """
    Draw images in a mosaic and save to output filename
    :param columns: number of columns in mosaic (default 5)
    :param rows: number of rows in mosaic (default 5)
    :param images_gen: generator of an array of images to contain in a mosaic
    :param output_filename: name of the file to save mosaic into
    """

    fig = plt.figure(num=None, figsize=(40, 40), dpi=80)

    for index, image in enumerate(images):
        a = fig.add_subplot(rows, columns, index + 1)
        plt.imshow(image)
        plt.axis("off")

    fig.tight_layout()
    fig.savefig(get_next_filename(output_filename))
    plt.close('all')


def transform_image(image):
    """
    :param image: original image
    :return: transformed image
    """

    transformable_image = rgb2grey(image)  # transformation beginning

    # median_transform = median(before_transform, disk(10))
    elevation_map = sobel(transformable_image)
    markers = np.ones_like(transformable_image)
    markers[transformable_image < 0.2] = 2
    markers[transformable_image > 0.5] = 0

    transformable_image = morphology.watershed(elevation_map, markers)

    segmentation = ndi.binary_fill_holes(transformable_image - 1)
    labeled, _ = ndi.label(segmentation)

    image_overlay = label2rgb(labeled, image=image, bg_label=0)

    return image_overlay


def transform(images):
    """
    :param images: array of images
    :return: transformed array of images
    """

    for index, image in enumerate(images):
        yield transform_image(image)


def main():
    plot_images(transform(get_images(IMAGE_DIR)), OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
