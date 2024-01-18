import tensorflow as tf
from keras import backend as K
from tensorflow.keras import layers


class Hswish(tf.keras.layers.Layer):
    def __init__(self):
        super(Hswish, self).__init__()

    def call(self, x):
        return x * tf.nn.relu6(x+3) / 6

class ShiftScaleLayer(tf.keras.layers.Layer):
    def __init__(self, shift, scale):
        super(ShiftScaleLayer, self).__init__()
        self.shift = shift
        self.scale = scale

    def build(self,input_shape):
        self._x = self.add_weight(name="shift",shape=(1),initializer=tf.constant_initializer(self.shift), trainable=True)
        self._y = self.add_weight(name="scale",shape=(1),initializer=tf.constant_initializer(self.scale), trainable=True)
        super(ShiftScaleLayer, self).build(input_shape)

    def call(self, x):
        x = tf.math.add(x, self._x)
        result = tf.math.multiply(x, self._y)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
def _inverted_res_block(inputs, kernel_size, expansion, stride, alpha, filters, block_id, activation="RE"):
    in_channels = K.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters*alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(expansion*in_channels, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        if activation == "RE":
            x = layers.ReLU(name=prefix + 'expand_relu')(x)
        else: 
            x = Hswish()(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=_correct_pad(x, (3, 3)), name=prefix + 'pad')(x)
        
    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                        strides=stride,
                        activation=None,
                        use_bias=False,
                        padding='same' if stride == 1 else 'valid',
                        name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    if activation == "RE":
        x = layers.ReLU(name=prefix + 'depthwise_relu')(x)
    else: 
        x = Hswish()(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                        kernel_size=1,
                        padding='same',
                        use_bias=False,
                        activation=None,
                        name=prefix + 'project')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def _correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
        # Arguments
            input_size: An integer or tuple/list of 2 integers.
            kernel_size - tuple: kernel size, (3,3)
        # Returns
            A tuple.
    """
    img_dim = 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def get_model(init_scale1, init_scale2, init_shift1, init_shift2):
    input1 = layers.Input(shape=(1, 128, 1))
    input2 = layers.Input(shape=(1, 128, 1))
    scaled1 = ShiftScaleLayer(init_shift1, init_scale1)(input1)
    scaled2 = ShiftScaleLayer(init_shift2, init_scale2)(input2)
    merged = layers.Concatenate(axis=2)([scaled1, scaled2])
    conv1 = layers.Conv2D(filters=8, kernel_size=(1, 3), strides=(1, 2))(merged)
    bn1 = layers.BatchNormalization()(conv1)

    x = Hswish()(bn1)
    x = _inverted_res_block(x, kernel_size=(1,3), filters=8, alpha=1, stride=1, expansion=1, activation="RE", block_id=0)
    x = _inverted_res_block(x, kernel_size=(1,3), filters=16, alpha=1, stride=1, expansion=3, activation="RE", block_id=1)
    x = _inverted_res_block(x, kernel_size=(1,3), filters=16, alpha=1, stride=1, expansion=3, activation="RE", block_id=2)
    x = _inverted_res_block(x, kernel_size=(1,3), filters=24, alpha=1, stride=1, expansion=3, activation="HS", block_id=3)
    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(2)(x)
    return tf.keras.models.Model(inputs=[input1, input2], outputs=output)

if __name__ == "__main__":
    model = get_model(0.1, 0.1, 0.1, 0.1)
    model.build((1, 12000, 1))
    model.summary()