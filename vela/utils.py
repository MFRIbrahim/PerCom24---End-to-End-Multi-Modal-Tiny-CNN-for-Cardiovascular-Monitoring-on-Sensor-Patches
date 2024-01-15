import tensorflow as tf


def set_tf_memory_growth():
    """
    When called, this function forces Tensorflow NOT to allocate every little piece of GPU memory there is. Instead, the
    amount of used GPU memory will be changed dynamically as needed. Nice when sharing a GPU resource.
    :return: N/A
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

class random_rep_data_gen:
    def __init__(self):
        self.size1 = [1, 1, 128, 1]
        self.size2 = [1, 1, 128, 1]
        self.name1 = "input_1"
        self.name2 = "input_2"

    def generator(self):
        for _ in range(100):
            yield {self.name1: tf.random.uniform(shape=self.size1), self.name2: tf.random.uniform(shape=self.size2)}
