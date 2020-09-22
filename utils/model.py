import tensorflow as tf
import numpy as np
import sys
import os
from utils.parse_config import parse_model_cfg

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def create_model(cfgpath, input_layer):

    model_summary = parse_model_cfg(cfgpath)

    all_layers = []
    feature_layers_index = []
    prev_layer = input_layer
    layer_cnt = 0
    
    for i, layer in enumerate(model_summary):
        if layer['type'] == 'convolutional':
            bn =  layer['batch_normalize']
            out_channels = layer['filters']
            k = layer['size']
            stride = layer['stride']
            pad = layer['pad']
            activation_type = layer['activation']

            if stride == 2:
                prev_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(prev_layer)
                padding = 'valid'
            elif stride == 1:
                padding = 'same'

            if bn == 1:
                batch_normalize = True
            elif bn == 0:
                batch_normalize = False

            if layer.get('groups'):
                group_num = layer['groups']
                in_channels = prev_layer.shape[-1]

                g = int(in_channels / group_num)
                conv_layers = []

                index = 0
                for i in range(g):
                    if not i == g - 1:
                        group = tf.keras.layers.Lambda(lambda x: x[:,:,:,index:index+g])(prev_layer)
                    else:
                        group = tf.keras.layers.Lambda(lambda x: x[:,:,:,index:])(prev_layer)
                    conv_layer_g = tf.keras.layers.Conv2D(filters = out_channels, 
                                                          kernel_size = k, 
                                                          strides = stride, 
                                                          padding = padding,
                                                          use_bias=not batch_normalize, 
                                                          kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                          bias_initializer=tf.constant_initializer(0.))(group)
                    conv_layers.append(conv_layer_g)
                    index += g

                conv_layer = tf.keras.concat(conv_layers)
            else:
                conv_layer = tf.keras.layers.Conv2D(filters = out_channels, 
                                                    kernel_size = k, 
                                                    strides = stride, 
                                                    padding = padding,
                                                    use_bias=not batch_normalize, 
                                                    kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                    bias_initializer=tf.constant_initializer(0.))(prev_layer)

            if batch_normalize: conv_layer = BatchNormalization()(conv_layer)

            if activation_type == "leaky":
                conv_layer = tf.nn.leaky_relu(conv_layer, alpha=0.1)
            elif activation_type == "mish":
                conv_layer = mish(conv_layer)

            all_layers.append(conv_layer)
            prev_layer = conv_layer
            layer_cnt += 1
        elif layer['type'] == 'route':
            get_layers = layer['layers']
            if len(get_layers) == 1:
                get_layer_index = get_layers[0]
                route_layer = all_layers[get_layer_index]
                all_layers.append(route_layer)
                prev_layer = route_layer
            else:
                route_layer = []
                for get_layer_index in get_layers:
                    route_layer.append(all_layers[get_layer_index])
                concat_layer = tf.concat(route_layer, axis=-1)
                all_layers.append(concat_layer)
                prev_layer = concat_layer
            layer_cnt += 1
        elif layer['type'] == 'shortcut':
            from_index = layer['from']
            shorcut_layer = prev_layer + all_layers[from_index[0]]
            all_layers.append(shorcut_layer)
            prev_layer = shorcut_layer
            layer_cnt += 1
        elif layer['type'] == 'maxpool':
            k = layer['size']
            stride = layer['stride']
            maxpool_layer = tf.nn.max_pool(prev_layer, ksize=k, padding='SAME', strides=stride)
            all_layers.append(maxpool_layer)
            prev_layer = maxpool_layer
            layer_cnt += 1
        elif layer['type'] == 'upsample':
            stride = layer['stride']
            upsample_layer = tf.image.resize(prev_layer, (prev_layer.shape[1] * stride, prev_layer.shape[2] * stride), method='bilinear')
            all_layers.append(upsample_layer)
            prev_layer = upsample_layer
            layer_cnt += 1
        elif layer['type'] == 'deconvolutional': # This layer denotes 'tf.keras.layers.Conv2DTranspose'
            # [deconvolutional]
            # filters=64
            # size=2
            # stride=2
            # activation=linear

            bn = False
            padding = 'valid'
            stride = layer['stride']
            k = layer['size']
            out_channels = layer['filters']
            activation_type = layer['activation']

            deconv_layer = tf.keras.layers.Conv2DTranspose(filters=out_channels,
                                                kernel_size = k, 
                                                strides=stride, 
                                                padding=padding, 
                                                activation=activation_type, 
                                                use_bias=not bn,
                                                kernel_regularizer=tf.keras.regularizers.l2(0.0005), 
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                bias_initializer=tf.constant_initializer(0.))(prev_layer)

            if bn: deconv_layer = BatchNormalization()(deconv_layer)

            all_layers.append(deconv_layer)
            prev_layer = deconv_layer
            layer_cnt += 1
        elif layer['type'] == 'yolo':
            feature_layers_index.append(len(all_layers)-1)
            all_layers.append(None)
            prev_layer = all_layers[-1]
        elif layer['type'] == 'net':
            pass
        else:
            raise OSError('Unsupported layer type!')

    return all_layers, feature_layers_index

if __name__ == '__main__':
    path = './cfg/yolov4.cfg'

    input_layer = tf.keras.Input(shape=(448, 448, 3))
    all_layers, features_layers_index = create_model(path,input_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=[all_layers[i] for i in features_layers_index])
    model.summary()

    for i in features_layers_index:
        print(all_layers[i])