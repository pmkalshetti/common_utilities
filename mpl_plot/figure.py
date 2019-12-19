import io
import matplotlib.pyplot as plt
import tensorflow as tf


def figure_to_image(fig):
    """Converts matplotlib figure to PNG image and closes the figure."""
    # save figure to PNG in memory
    buf = io.BytesIO()
    fig.savefig(buf, format="png")

    # closing the figure prevents it from directly displaying in notebook
    plt.close(fig)

    # convert PNG buffer to TF image
    buf.seek(0)
    img = tf.image.decode_png(buf.getvalue(), channels=4)

    return img


def image_to_figure(img):
    """Plots image on figure and returns the figure."""
    # remove batch axis
    if img.ndim == 4:
        assert img.shape[0] == 1, "Multiple images in batch not allowed."
        img = img[0]

    # if grayscale, then squeeze last dim and set cmap
    if img.shape[-1] == 1:
        img = img[..., 0]
        cmap = "bwr"
        vmin = -1
        vmax = 1
    else:
        cmap = None
        vmin = None
        vmax = None

    fig = plt.figure()
    plt.imshow(img, cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")

    return fig


def make_loggable_image_plot(img):
    """Plots image and returns as tensor with batch axis for logging."""
    fig = image_to_figure(img)

    img = figure_to_image(fig)

    # add batch dim for logging
    img = img[tf.newaxis, ...]

    return img
