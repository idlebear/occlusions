from matplotlib import widgets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


def create_gaussian(height, width, origin=(0, 0), centre=(0, 0), sigma=1, scale=1):
    """create_2d_gaussian -- creates a 2D gaussian kernal at the
           specified offset, and scale

    @param size     the size of the gaussian kernel to create
    @param centre   the mean(mu) location of the gaussian.
    @param sigma    sigma -- the standard deviation of the curve
    @param scale    numerical width of each step
    """
    x, y = np.meshgrid(
        np.arange(
            origin[0] - centre[0],
            origin[0] - centre[0] + width * scale,
            scale,
        ),
        np.arange(
            origin[1] - centre[1],
            origin[1] - centre[1] + width * scale,
            scale,
        ),
        indexing="xy",
    )

    # since the gaussian may be off-center (and is definitely truncated), normalize
    # using the sum of elements
    gus = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * sigma * sigma))
    return gus / np.sum(gus)


def clip_gaussian(gaussian):
    return np.clip(gaussian, 0, 1)


def normalize_gaussian(gaussian):
    return gaussian / np.sum(gaussian)


# brief validation code
if __name__ == "__main__":

    scale = 0.03
    size = 50
    centre = (1.0, 1.0)
    origin = (0.0, 0.0)
    sigma = 0.1
    g = create_gaussian(height=size, width=size, origin=origin, centre=centre, sigma=sigma, scale=scale)

    # scale = 0.05
    # size = 300
    # centre = (4,12)
    # origin = (2,2)
    # sigma = 3
    # g2 = create_gaussian(height=size, width=size, origin=origin,  centre=centre, sigma=sigma, scale=scale)

    # g = g + g2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xv, yv = np.meshgrid(range(len(g)), range(len(g)))
    xv = origin[0] + xv * scale
    yv = origin[1] + yv * scale
    ax.plot_wireframe(xv, yv, g)

    plt.show()

    print("done")
