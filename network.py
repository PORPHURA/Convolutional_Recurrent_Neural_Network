from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import ops

from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow.keras as k
from tensorflow.keras.optimizers import Adam
from convolutional_recurrent import ConvGRU2D

# float16 computing precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

class Generator(k.Model):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.loss_type = config.g_loss_type
        self.loss_wt = list(config.loss_wt.values())
        self.n_levels = config.n_levels
        self.batch_size = config.batch_size

        net_params = {
            'down_ch_conv': [[config.n_ch//2, config.n_ch]] + 
                [
                    [2**(i-1)*config.n_ch, 2**i*config.n_ch] for i in range(1, self.n_levels)
                ],
            'down_ker_conv': [[config.ker_size, config.ker_size]] * self.n_levels,
            'ch_rconv': [
                    [2**i*config.n_ch, 2**i*config.n_ch] for i in range(self.n_levels)
                ],
            'ker_rconv': [[config.ker_size, config.ker_size]] * self.n_levels,
            'up_ch_conv': [
                [2**(self.n_levels-1-i)*config.n_ch, 2**(self.n_levels-2-i)*config.n_ch] for i in range(self.n_levels-1)
                ] + 
                [[config.n_ch, config.n_ch]],
            'up_ker_conv': [[config.ker_size, config.ker_size]] * self.n_levels
        }

        self.in_conv1 = k.layers.Conv2D(int(config.n_ch / 2), config.ker_size, strides=1, padding='same', activation='relu')
        self.in_conv2 = k.layers.Conv2D(config.n_ch, config.ker_size, strides=1, padding='same', activation='relu')
        # down-sampling path
        self.down = {}
        self.mp = {}
        self.skip = {}  # skip connection
        for i, (ker_conv, ch_conv, ker_rconv, ch_rconv) in enumerate(zip(net_params['down_ker_conv'], net_params['down_ch_conv'], net_params['ker_rconv'], net_params['ch_rconv'])):
            self.down[str(i)] = DownBlock(ker_conv, ker_rconv, ch_conv, ch_rconv)  # string key is required by Tensorflow
            self.mp[str(i)] = k.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')
            self.skip[str(i)] = RCBlock(ker_rconv, ch_rconv)

        # the middle block
        ch = 2**(self.n_levels)*config.n_ch
        self.mid_conv = DownBlock([config.ker_size], [config.ker_size], [ch], [ch])
        self.mid_rconv = RCBlock([config.ker_size], [ch])

        # the up-sampling path
        self.up = {}
        self.deconv = {}
        for i, (ker_conv, ch_conv) in enumerate(zip(net_params['up_ker_conv'], net_params['up_ch_conv'])):
            self.up[str(i)] = UpBlock(ker_conv, ch_conv)
            self.deconv[str(i)] = k.layers.UpSampling2D(size=2, interpolation='nearest')

        # one more convolutional block
        self.out_conv1 = k.layers.Conv2D(int(config.n_ch / 2), config.ker_size, strides=1, padding='same', activation='relu')
        self.out_conv2 = k.layers.Conv2D(2, config.ker_size, strides=1, padding='same', activation='linear')  # linear activation

    def call(self, inp):
        assert len(self.down)==len(self.up), 'The lengths of Down path and Up path do not match'
        
        # input conv
        t = inp.shape[1]
        x_in = tf.reshape(inp, [inp.shape[0]*inp.shape[1], inp.shape[2], inp.shape[3], inp.shape[4]])
        x_in = self.in_conv1(x_in)
        x_in = self.in_conv2(x_in)

        # down-sampling path
        x_down = {}  # output of each down block
        x_br = {}  # bridge between down and up blocks
        x_ds = x_in  # input of next down block
        for i in range(self.n_levels):
            # downsampling convolution block
            x_down[str(i)] = self.down[str(i)](x_ds)
            # skip connection with recurrent convolution block
            x_br[str(i)] = self.skip[str(i)](x_down[str(i)], self.batch_size)
            # max-pooling
            x_ds = self.mp[str(i)](x_down[str(i)])

        # middle block
        x_mid = self.mid_conv(x_ds)
        x_mid = self.mid_rconv(x_mid, self.batch_size)

        # up-sampling path
        x_up= {}  # output of each up block
        x_up['-1'] = x_mid[t-1::t,:,:,:]
        # x_up['-1'] = tf.reshape(x_mid, [inp.shape[0]*inp.shape[1], x_mid.shape[2], x_mid.shape[3], x_mid.shape[4]])
        for i in range(self.n_levels):
            x_us = self.deconv[str(i)](x_up[str(i-1)])
            x_up[str(i)] = self.up[str(i)](x_us, x_br[str(self.n_levels-1-i)][t-1::t,:,:,:])

        # output conv
        x_out = self.out_conv1(x_up[str(self.n_levels-1)])
        out = self.out_conv2(x_out)
        # out = tf.reshape(x_out, [inp.shape[0], inp.shape[1], x_out.shape[1], x_out.shape[2], x_out.shape[3]])
        return out  # return the whole sequence and the last output

    def inference(self, input_batch):
        # infer the generator output
        return self.call(input_batch)

    def train(self, tape, loss, optimizer):
        grad = tape.gradient(loss, self.trainable_variables)
        # for g in grad:
        #     tf.debugging.check_numerics(g, message='Invalid value break')  # check for NAN & Inf
        optimizer.apply_gradients(zip(grad, self.trainable_variables))

    def loss(self, output_batch, logit_fake, logit_real, target_batch):
        output_batch = tf.cast(output_batch, dtype=target_batch.dtype)
        dis_loss = tf.reduce_mean(tf.keras.losses.MSE(logit_fake, tf.ones_like(logit_fake)))
        # berhu_loss = tf.reduce_mean(ops.berhu_loss(labels=target_batch, predictions=output_batch, delta=0.1, adaptive=False))
        mae_loss = tf.reduce_mean(k.losses.MAE(target_batch, output_batch))
        ssim_loss = tf.reduce_mean(1 - tf.image.ssim_multiscale(target_batch, output_batch, max_val=2))
        loss = [dis_loss, mae_loss, ssim_loss]
        total_loss = tf.reduce_sum(tf.multiply(self.loss_wt, loss))
        loss.append(total_loss)
        return loss


class UpBlock(k.layers.Layer):
    def __init__(self, ker_conv, ch_conv):
        super(UpBlock, self).__init__()
        self.conv = []
        self.bn =[]
        self.relu = []
        for ker_size, n_ch in zip(ker_conv, ch_conv):
            self.conv.append(k.layers.Conv2D(filters=n_ch, kernel_size=ker_size, strides=1, padding='same'))
            self.bn.append(k.layers.BatchNormalization(axis=-1))
            self.relu.append(k.layers.LeakyReLU())

    def __call__(self, inp, cat):
        x_act = tf.concat([inp, cat], axis=-1)
        for conv_layer, bn_layer, relu_layer in zip(self.conv, self.bn, self.relu):
            x_conv = conv_layer(x_act)
            x_bn = bn_layer(x_conv)
            x_act = relu_layer(x_bn)
        return x_act


class RCBlock(k.layers.Layer):
    def __init__(self, ker_rconv, ch_rconv):
        super(RCBlock, self).__init__()
        self.rconv = []
        for ker_size, n_ch in zip(ker_rconv, ch_rconv):
            rconv = ConvGRU2D(filters=n_ch, kernel_size=ker_size, strides=1, padding='same', return_sequences=True)
            rconv.trainable = False
            self.rconv.append(rconv)
        self.conv = k.layers.Conv2D(filters=n_ch, kernel_size=1, strides=1, padding='same')
        self.conv.trainable = False

    def __call__(self, inp, n):
        assert inp.ndim == 4, 'Inconsistent dimensions'
        inp_shp = inp.shape
        x_rconv = tf.reshape(inp, [n, inp_shp[0]//n, inp_shp[1], inp_shp[2], inp_shp[3]])
        for rconv_layer in self.rconv:
            x_rconv = rconv_layer(x_rconv)
        x_conv = tf.reshape(x_rconv, inp_shp)
        x_conv = self.conv(x_conv)
        return x_conv + inp


class DownBlock(k.layers.Layer):
    def __init__(self, ker_conv, ker_rconv, ch_conv, ch_rconv):
        super(DownBlock, self).__init__()
        self.conv = []
        self.bn = []
        self.relu = []
        for ker_size, n_ch in zip(ker_conv, ch_conv):
            self.conv.append(k.layers.Conv2D(filters=n_ch, kernel_size=ker_size, strides=1, padding='same'))
            self.bn.append(k.layers.BatchNormalization(axis=-1))
            self.relu.append(k.layers.LeakyReLU())
    
    def __call__(self, inp):
        x_act = inp
        for conv_layer, bn_layer, relu_layer in zip(self.conv, self.bn, self.relu):
            x_conv = conv_layer(x_act)
            x_bn = bn_layer(x_conv)
            x_act = relu_layer(x_bn)
        return x_act  # output next down block


class Discriminator(k.Model):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.n_blocks = config.n_blocks

        # the first block
        self.conv1 = k.layers.Conv2D(config.n_ch, config.ker_size, strides=1, padding='same', activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros')

        # the normal blocks
        self.nb = dict()
        for i in range(self.n_blocks):
            self.nb[str(i+1)] = normal_block(config.ker_size, 2**(i+1)*config.n_ch)

        # average pooling on the spatial direction
        self.ap = k.layers.GlobalAveragePooling2D()

        # fully connected layers
        self.fc1 = k.layers.Dense(config.n_ch, activation='relu', kernel_initializer='glorot_normal', bias_initializer='zeros')
        self.fc2 = k.layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='zeros', dtype='float32')

    def call(self, inp):
        x = dict()
        x['0'] = self.conv1(inp)
        for i in range(self.n_blocks):
            x[str(i+1)] = self.nb[str(i+1)](x[str(i)])
        x_ap = self.ap(x[str(self.n_blocks)])
        x_fc1 = self.fc1(x_ap)
        x_fc2 = self.fc2(x_fc1)
        return x_fc2

    def inference(self, output_batch, target_batch):
        # infer the generator output
        logit_fake = self.call(output_batch)
        logit_real = self.call(target_batch)
        return logit_fake, logit_real

    def train(self, tape, loss, optimizer):
        grad = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grad, self.trainable_variables))

    def loss(self, logit_fake, logit_real):
        fake_loss = tf.reduce_mean(tf.keras.losses.MSE(tf.zeros_like(logit_fake), logit_fake))
        real_loss = tf.reduce_mean(tf.keras.losses.MSE(tf.ones_like(logit_real), logit_real))
        total_loss = (fake_loss + real_loss) / 2
        return [fake_loss, real_loss, total_loss]

class normal_block(k.layers.Layer):
    def __init__(self, ker_size, n_ch, downsample=True):
        super(normal_block, self).__init__()
        if not isinstance(n_ch, list):
            n_ch = [n_ch, 2*n_ch]
        self.conv1 = k.layers.Conv2D(n_ch[0], ker_size, strides=1, padding='same', activation='relu', kernel_initializer='glorot_normal')
        if downsample:
            self.conv2 = k.layers.Conv2D(n_ch[1], ker_size, strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal')
        else:
            self.conv2 = k.layers.Conv2D(n_ch[1], ker_size, strides=1, padding='same', activation='relu', kernel_initializer='glorot_normal')

    def __call__(self, inp):
        x_conv1 = self.conv1(inp)
        x_conv2 = self.conv2(x_conv1)
        return x_conv2
