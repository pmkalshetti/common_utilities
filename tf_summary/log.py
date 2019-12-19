import tensorflow as tf
from common_utilities.instance import make_list


def log_summary(writer, epoch,
                scalars=None, str_scalars=None,
                images=None, str_images=None):
    """Writes scalar and image summary for logging."""
    assert (scalars is not None) or (images is not None), \
        "Atleast one of scalars or images must be present along with their str"

    with writer.as_default():
        if scalars is not None:
            scalars = make_list(scalars)  # handles non list argument
            str_scalars = make_list(str_scalars)
            assert len(scalars) == len(str_scalars)
            for str_scalar, scalar in zip(str_scalars, scalars):
                tf.summary.scalar(str_scalar, scalar, step=epoch)

        if images is not None:
            images = make_list(images)
            str_images = make_list(str_images)
            assert len(images) == len(str_images)
            for str_image, image in zip(str_images, images):
                tf.summary.image(str_image, image, step=epoch)

    return
