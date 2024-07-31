import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import InputSpec
from keras.layers import Layer
from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.layers import Add
from keras.layers import Dropout
from keras.backend import image_data_format


def res_block(input, filters, kernel_size=(3, 3), strides=(1, 1), use_dropout=False):
    """
    Instanciate a Keras Resnet Block using sequential API.

    :param input: Input tensor
    :param filters: Number of filters to use
    :param kernel_size: Shape of the kernel for the convolution
    :param strides: Shape of the strides for the convolution
    :param use_dropout: Boolean value to determine the use of dropout
    :return: Keras Model
    """
    x = ReflectionPadding2D((1, 1))(input)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    x = ReflectionPadding2D((1, 1))(x)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,)(x)
    x = BatchNormalization()(x)

    merged = Add()([input, x])
    return merged


def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """
    Pad the 2nd and 3rd dimensions of a 4D tensor.

    :param x: Input tensor
    :param padding: Shape of padding to use
    :param data_format: Tensorflow vs Theano convention ('channels_last', 'channels_first')
    :return: Tensorflow tensor
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if data_format == 'channels_first':
        pattern = [[0, 0],
                   [0, 0],
                   list(padding[0]),
                   list(padding[1])]
    else:
        pattern = [[0, 0],
                   list(padding[0]), list(padding[1]),
                   [0, 0]]
    return tf.pad(x, pattern, "REFLECT")


# class ReflectionPadding2D(Layer):
#     """Reflection-padding layer for 2D input (e.g. picture).
#     This layer can add rows and columns or zeros
#     at the top, bottom, left and right side of an image tensor.
#     # Arguments
#         padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
#             - If int: the same symmetric padding
#                 is applied to width and height.
#             - If tuple of 2 ints:
#                 interpreted as two different
#                 symmetric padding values for height and width:
#                 `(symmetric_height_pad, symmetric_width_pad)`.
#             - If tuple of 2 tuples of 2 ints:
#                 interpreted as
#                 `((top_pad, bottom_pad), (left_pad, right_pad))`
#         data_format: A string,
#             one of `channels_last` (default) or `channels_first`.
#             The ordering of the dimensions in the inputs.
#             `channels_last` corresponds to inputs with shape
#             `(batch, height, width, channels)` while `channels_first`
#             corresponds to inputs with shape
#             `(batch, channels, height, width)`.
#             It defaults to the `image_data_format` value found in your
#             Keras config file at `~/.keras/keras.json`.
#             If you never set it, then it will be "channels_last".
#     # Input shape
#         4D tensor with shape:
#         - If `data_format` is `"channels_last"`:
#             `(batch, rows, cols, channels)`
#         - If `data_format` is `"channels_first"`:
#             `(batch, channels, rows, cols)`
#     # Output shape
#         4D tensor with shape:
#         - If `data_format` is `"channels_last"`:
#             `(batch, padded_rows, padded_cols, channels)`
#         - If `data_format` is `"channels_first"`:
#             `(batch, channels, padded_rows, padded_cols)`
#     """

#     def __init__(self,
#                  padding=(1, 1),
#                  data_format=None,
#                  **kwargs):
#         super(ReflectionPadding2D, self).__init__(**kwargs)
#         self.data_format = image_data_format()
#         if isinstance(padding, int):
#             self.padding = ((padding, padding), (padding, padding))
#         elif hasattr(padding, '__len__'):
#             if len(padding) != 2:
#                 raise ValueError("Padding should have two elements. Found: " + str(padding))

#             # Check and adjust padding for the height dimension
#             if isinstance(padding[0], int):
#                 height_padding = (padding[0], padding[0])
#             elif len(padding[0]) == 1:
#                 height_padding = (padding[0][0], padding[0][0])
#             elif len(padding[0]) == 2:
#                 height_padding = padding[0]
#             else:
#                 raise ValueError("Invalid padding format for the height dimension")

#             # Check and adjust padding for the width dimension
#             if isinstance(padding[1], int):
#                 width_padding = (padding[1], padding[1])
#             elif len(padding[1]) == 1:
#                 width_padding = (padding[1][0], padding[1][0])
#             elif len(padding[1]) == 2:
#                 width_padding = padding[1]
#             else:
#                 raise ValueError("Invalid padding format for the width dimension")

#             # Now height_padding and width_padding contain the normalized padding
#             self.padding = (height_padding, width_padding)
#         else:
#             raise ValueError('`padding` should be either an int, '
#                              'a tuple of 2 ints '
#                              '(symmetric_height_pad, symmetric_width_pad), '
#                              'or a tuple of 2 tuples of 2 ints '
#                              '((top_pad, bottom_pad), (left_pad, right_pad)). '
#                              'Found: ' + str(padding))
#         self.input_spec = InputSpec(ndim=4)

#     def compute_output_shape(self, input_shape):
#         if self.data_format == 'channels_first':
#             if input_shape[2] is not None:
#                 rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
#             else:
#                 rows = None
#             if input_shape[3] is not None:
#                 cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
#             else:
#                 cols = None
#             return (input_shape[0],
#                     input_shape[1],
#                     rows,
#                     cols)
#         elif self.data_format == 'channels_last':
#             if input_shape[1] is not None:
#                 rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
#             else:
#                 rows = None
#             if input_shape[2] is not None:
#                 cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
#             else:
#                 cols = None
#             return (input_shape[0],
#                     rows,
#                     cols,
#                     input_shape[3])

#     def call(self, inputs):
#         return spatial_reflection_2d_padding(inputs,
#                                              padding=self.padding,
#                                              data_format=self.data_format)

#     def get_config(self):
#         config = {'padding': self.padding,
#                   'data_format': self.data_format}
#         base_config = super(ReflectionPadding2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))



# input = Input(shape=(256, 256, 3))
# x = ReflectionPadding2D(3)(input)
# model = Model(input, x)
# model.summary()


def normalize_data_format(data_format):
    """
    Normalize the data_format argument to one of the recognized formats.

    Args:
        data_format: str, the format to normalize

    Returns:
        str, normalized format
    """
    if data_format is None:
        data_format = 'channels_last'
    data_format = str(data_format).lower()
    if data_format not in {'channels_last', 'channels_first'}:
        raise ValueError("Invalid data_format, must be 'channels_last' or 'channels_first'.")
    return data_format


def normalize_tuple(value, n, name):
    """
    Normalize the value to a tuple of length n.

    Args:
        value: int or tuple of int
        n: int, the desired length of the tuple
        name: str, the name of the parameter for error messages

    Returns:
        tuple of length n
    """
    if isinstance(value, tuple):
        if len(value) != n:
            raise ValueError(f"Invalid {name} length, should be {n}.")
        return value
    elif isinstance(value, int):
        return (value,) * n
    else:
        raise ValueError(f"Invalid {name}, should be int or tuple of {n} ints.")


class ReflectionPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).
    This layer can add rows and columns or zeros
    at the top, bottom, left and right side of an image tensor.
    # Arguments
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to width and height.
            - If tuple of 2 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints:
                interpreted as
                `((top_pad, bottom_pad), (left_pad, right_pad))`
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, padded_rows, padded_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, padded_rows, padded_cols)`
    """

    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    input_shape[1],
                    rows,
                    cols)
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])

    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs,
                                             padding=self.padding,
                                             data_format=self.data_format)

    def get_config(self):
        config = {'padding': self.padding,
                  'data_format': self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))