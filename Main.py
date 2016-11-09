import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray

def main():
    image = data.chelsea()

    image = rgb2hsv(image)
    image[:,:,1] = 0
    image = hsv2rgb(image)

    plt.imshow(image)
    plt.savefig('my_image.jpg')

if __name__ == "__main__":
    main()