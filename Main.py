import matplotlib.pyplot as plt
import numpy as np

def main() -> null:
    x = np.array([1, 2, 3, 4, 5])
    y = x * 20

    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()