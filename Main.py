import matplotlib.pyplot as plt
import os
from skimage import data

IMAGE_DIR = 'Images/'
OUTPUT_FILENAME = 'result.jpg'
ROW_NUM = 2


def get_images():
    images_filenames = os.listdir(IMAGE_DIR)
    images = []

    for image_filename in images_filenames:
        abs_image_filename = os.path.abspath(os.path.join(IMAGE_DIR, image_filename))
        images.append(data.imread(abs_image_filename, as_grey=True))

    return images


def plot_images(images):
    fig = plt.figure(num=None, figsize=(40, 40), dpi=80)

    for index, image in enumerate(images):
        a = fig.add_subplot(5, 5, index + 1)
        plt.imshow(image, cmap="Greys_r")
        a.set_title(str(index))
        plt.axis("off")

    fig.savefig(OUTPUT_FILENAME)


def transform_image(image):
    after_transform = image
    return after_transform


def transform(images):
    for index, image in enumerate(images):
        images[index] = transform_image(image)
    return images


def main():
    plot_images(transform(get_images()))


if __name__ == "__main__":
    main()
