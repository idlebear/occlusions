import numpy as np


def create_gaussian(size, origin=(0, 0), centre=(0, 0), sigma=1, scale=1):
    """create_gaussian -- creates a 2D gaussian kernal at the
           specified offset, and scale

    @param size      the number of steps in the gaussian kernel
    @param origin    the origin of the grid
    @param centre    the offset from the origin of the peak value in X and Y.
    @param sigma     sigma -- the standard deviation of the curve
    @param scale     numerical width of each step

    @return a 2D numpy array representing the gaussian kernel
    """

    x, y = np.meshgrid(
        np.arange(
            origin[0] - centre[0],
            origin[0] - centre[0] + size * scale,
            scale,
        ),
        np.arange(
            origin[1] - centre[1],
            origin[1] - centre[1] + size * scale,
            scale,
        ),
        indexing="xy",
    )

    # since the gaussian may be off-center (and is definitely truncated), normalize
    # using the sum of elements
    gus = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * sigma * sigma))
    return gus / np.sum(gus)


# brief validation code
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    scale = 1
    size = 5
    centre = (0.0, 0.0)
    origin = (-(size // 2), -(size // 2))
    sigma = 1
    g = create_gaussian(
        size=size, origin=origin, centre=centre, sigma=sigma, scale=scale
    )

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # xv, yv = np.meshgrid(range(len(g)), range(len(g)))
    # xv = origin[0] + xv * scale
    # yv = origin[1] + yv * scale
    # ax.plot_wireframe(xv, yv, g)

    plt.imshow(g, cmap="plasma")
    plt.colorbar()
    plt.savefig("gaussian_test.png")
