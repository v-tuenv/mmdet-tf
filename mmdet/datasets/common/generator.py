import tensorflow as tf
def make_generator(seed=None):
  """Creates a random generator.
  Args:
    seed: the seed to initialize the generator. If None, the generator will be
      initialized non-deterministically.
  Returns:
    A generator object.
  """
  if seed is not None:
    return tf.random.Generator.from_seed(seed)
  else:
    return tf.random.Generator.from_non_deterministic_state()