from tensorflow.python import tf2  # pylint: disable=import-outside-toplevel


def is_tf1():
  """Whether current TensorFlow Version is 1.X."""
  return not tf2.enabled()


def is_tf2():
  """Whether current TensorFlow Version is 2.X."""
  return tf2.enabled()