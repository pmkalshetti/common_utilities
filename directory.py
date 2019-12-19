import tensorflow as tf


def create_dir(path: str, flag_delete_existing: bool) -> bool:
    """Returns True if created new dir, else False."""
    if flag_delete_existing and tf.io.gfile.exists(path):
        tf.io.gfile.rmtree(path)
    if not tf.io.gfile.exists(path):
        tf.io.gfile.makedirs(path)
    else:
        return False

    return True
