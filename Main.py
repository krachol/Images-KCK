import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.color import rgb2hsv, hsv2rgb,

def main():
    image = data.chelsea()

    fig = plt.figure()
    a = fig.add_subplot(2,1,1)

    plt.imshow(image)

    a = fig.add_subplot(2,1,2)

    image = rgb2hsv(image)
    image[:,:,1] = 0
    image = hsv2rgb(image)
    plt.imshow(image)


    fig.savefig('my_image.jpg')

if __name__ == "__main__":
    main()