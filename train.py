from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import shutil
import glob
import numpy as np
import scipy.io
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from configobj import ConfigObj
from time import time
import tensorflow as tf
import network
from ops import *

# global parameters
n_channels = 16
n_levels = 5
n_blocks = 5

g_loss_type = ['dis', 'mae', 'ssim']
d_loss_type = ['fake', 'real']
loss_wt = {'dis': 0.3, 'mae': 3, 'ssim': 1}

lr_g, lr_d = 1e-5, 1e-6     # learning rate for generator and discriminator respectively

n_epoch = 120
n_train_batch = 100  # number of training batches before validating
n_valid_batch = 20  # number of batches for each validation
n_log = 20  # training batch number before writing a log


# read the write directories
rd_root =   ###### NEED TO FILL IN DATASET PATH FOR PRE-TRAINING
# model_name = 'RH_M_chn=20_mae=3.0_ssim=1.0_lr=5.0e-05-1.0e-06_n=5_trial=mix'
model_name = 'RH_M_chn=%d_mae=%.1f_ssim=%.1f_lr=%.1e-%.1e_n=%d_trial=scratch' % (n_channels, loss_wt['mae'], loss_wt['ssim'], lr_g, lr_d, n_plane)  # the post-fix for tensorboard and model directory
print('model:{}'.format(model_name))
model_dir = os.path.join('Models', model_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()
    tc.batch_size, vc.batch_size = 3, 3
    tc.ker_size, vc.ker_size = 3, 3
    tc.c_leaky, vc.c_leaky = 0.1, 0.1
    tc.n_ch, vc.n_ch = n_channels, n_channels
    tc.n_blocks, vc.n_blocks = n_blocks, n_blocks
    tc.n_levels, vc.n_levels = n_levels, n_levels
    tc.buf_size, vc.buf_size = 32, 32
    tc.g_loss_type, vc.g_loss_type = g_loss_type, g_loss_type
    tc.d_loss_type, vc.d_loss_type = d_loss_type, d_loss_type
    tc.loss_wt, vc.loss_wt = loss_wt, loss_wt
    return tc, vc

def train_step(input_seq, target_batch):
    # GAN training step
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        # generator forward inference
        output_batch = gen.inference(input_seq)
        # discriminator forward inference
        logit_fake, logit_real = dis.inference(output_batch, target_batch)
        # compute loss
        g_loss = gen.loss(output_batch, logit_fake, logit_real, target_batch)
        d_loss = dis.loss(logit_fake, logit_real)
    # train parameters
    gen.train(g_tape, g_loss[-1], g_opt)
    dis.train(d_tape, d_loss[-1], d_opt)

    return output_batch, g_loss, d_loss

def valid_step(input_seq, target_batch):
    # GAN evaluating step
    
    # generator forward inference
    output_batch = gen.inference(input_seq)
    # discriminator forward inference
    logit_fake, logit_real = dis.inference(output_batch, target_batch)
    # compute loss
    g_loss = gen.loss(output_batch, logit_fake, logit_real, target_batch)
    d_loss = dis.loss(logit_fake, logit_real)

    return output_batch, [l.numpy() for l in g_loss], [l.numpy() for l in d_loss]

def write_summary(input_seq, output_batch, target_batch, g_loss, d_loss, name_scope, step):
    with writer.as_default():
        with tf.name_scope(name_scope):
            for i_seq in range(input_seq.shape[1]):
                tf.summary.image('image_input_real'+str(i_seq), np.array([min_max_norm(comp2ap(inp[i_seq,:,:,:])[:,:,0:1]) for inp in input_seq.numpy()]), step=step, max_outputs=2)
                tf.summary.image('image_input_imag'+str(i_seq), np.array([min_max_norm(comp2ap(inp[i_seq,:,:,:])[:,:,1:2]) for inp in input_seq.numpy()]), step=step, max_outputs=2)
            output_ap = comp2ap(output_batch.numpy())
            target_ap = comp2ap(target_batch.numpy())
            tf.summary.image('image_output_amp', np.array([min_max_norm(out_amp) for out_amp in output_ap[:,:,:,0:1]]), step=step, max_outputs=2)
            tf.summary.image('image_output_ph', np.array([min_max_norm(out_ph) for out_ph in output_ap[:,:,:,1:2]]), step=step, max_outputs=2)
            tf.summary.image('image_target_amp', np.array([min_max_norm(tar_amp) for tar_amp in target_ap[:,:,:,0:1]]), step=step, max_outputs=2)
            tf.summary.image('image_target_ph', np.array([min_max_norm(tar_ph) for tar_ph in target_ap[:,:,:,1:2]]), step=step, max_outputs=2)
        with tf.name_scope(name_scope + 'Generator'):
            for typ in range(len(g_loss_type)):
                tf.summary.scalar(g_loss_type[typ]+'_loss', g_loss[typ], step=step)
            tf.summary.scalar('total_loss', g_loss[-1], step=step)
        with tf.name_scope(name_scope + 'Discriminator'):
            for typ in range(len(d_loss_type)):
                tf.summary.scalar(d_loss_type[typ]+'_loss', d_loss[typ], step=step)
            tf.summary.scalar('total_loss', d_loss[-1], step=step)


if __name__ == '__main__':

    # initialize parameters
    t_config, v_config = init_parameters()

    with tf.device('/cpu:0'):

        # generate train & valid dataset
        train_data = glob.glob(rd_root + '/train/*.mat')
        valid_data = glob.glob(rd_root + '/valid/*.mat')
        random.shuffle(train_data)
        random.shuffle(valid_data)
        if not train_data:
            print('Converting training data to numpy file...')
            train_data = mat2npz(glob.glob(rd_root + '/train/*.mat'))
        if not valid_data:
            print('Converting validation data to numpy file...')
            valid_data = mat2npz(glob.glob(rd_root + '/valid/*.mat'))
        print('Training = %d, Validation = %d' % (len(train_data)*n_sample, len(valid_data)*n_sample))

        # load the data
        print('Loading data...')
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.interleave(lambda x: 
        tf.data.Dataset.from_tensor_slices(tuple(
            tf.py_function(load_data, inp=[x], Tout=((tf.float32,tf.float32))))
        ), 
        cycle_length=t_config.batch_size, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
        valid_dataset = valid_dataset.interleave(lambda x: 
        tf.data.Dataset.from_tensor_slices(tuple(
            tf.py_function(load_data, inp=[x], Tout=((tf.float32,tf.float32))))
        ), 
        cycle_length=v_config.batch_size, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # augment dataset
        # print('Augmenting data...')
        # train_dataset = dataset_augment(train_dataset)
        # shuffle & batch
        train_dataset = train_dataset.shuffle(t_config.buf_size, reshuffle_each_iteration=True).batch(t_config.batch_size, drop_remainder=True).prefetch(t_config.buf_size)
        valid_dataset = valid_dataset.repeat(int(n_epoch*n_valid_batch/len(valid_data)) + 1).shuffle(v_config.buf_size).batch(v_config.batch_size, drop_remainder=True).prefetch(v_config.buf_size)

    with tf.device('/gpu:0'):  # change to cpu:0 if no GPU available

        # create Generator & Discriminator
        gen = network.Generator(t_config)
        dis = network.Discriminator(t_config)
        # create optimizers
        g_opt = tf.keras.optimizers.Adam(lr_g, epsilon=1e-6)
        d_opt = tf.keras.optimizers.Adam(lr_d, epsilon=1e-6)

        # create checkpoint
        ckpt = tf.train.Checkpoint(gen=gen, dis=dis, g_opt = g_opt, d_opt=d_opt)
        ckpt_mgr = tf.train.CheckpointManager(ckpt, directory=model_dir, max_to_keep=2, checkpoint_name='best_model')
        ckpt_berhu = tf.train.CheckpointManager(ckpt, directory=model_dir, max_to_keep=2, checkpoint_name='best_berhu')
        ckpt_ssim = tf.train.CheckpointManager(ckpt, directory=model_dir, max_to_keep=2, checkpoint_name='best_ssim')
        if ckpt_mgr.latest_checkpoint:
            ckpt.restore(ckpt_mgr.latest_checkpoint)
            print('Model loaded successfully')
        else:
            print('No model available')

        # create tensorboard log
        if os.path.exists("log/" + model_name):
            shutil.rmtree("log/" + model_name)
        writer = tf.summary.create_file_writer('log/'+model_name)
        start_time = time()
        min_loss, min_berhu, min_ssim = float('inf'), float('inf'), float('inf')

        step = 0
        for epoch in range(n_epoch):
            print('Epoch: %d\n' % epoch)

            for t_batch, (t_input_seq, t_target) in train_dataset.enumerate():
                step += 1

                # training step
                t_output, g_loss, d_loss = train_step(input_seq=t_input_seq, target_batch=t_target)

                # write to tensorboard
                if t_batch % n_log == 0:
                    write_summary(t_input_seq, t_output, t_target, g_loss, d_loss, 'Training', step)
                    print('Training: iter %d in epoch %d time: %.2f || train_G_loss: %.4f train_D_loss: %.4f' % (t_batch+1, epoch, time()-start_time, g_loss[-1], d_loss[-1]))
                
                # validate every #checkpoint batches
                if t_batch % n_train_batch == 0:
                    valid_g_loss = np.zeros([len(g_loss_type)+1,], dtype='float32')
                    valid_d_loss = np.zeros([len(d_loss_type)+1,], dtype='float32')
                    # validate over #n_valid_batch batches
                    for (v_input_seq, v_target) in valid_dataset.take(n_valid_batch):
                        v_output, v_g_loss, v_d_loss = valid_step(input_seq=v_input_seq, target_batch=v_target)
                        valid_g_loss += v_g_loss
                        valid_d_loss += v_d_loss
                    
                    # average
                    valid_g_loss /= n_valid_batch
                    valid_d_loss /= n_valid_batch

                    # write to tensorboard
                    fmt_str = '\nValidating: iter %d in epoch %d time: %.2f || G_loss: %.4f D_loss: %.4f || train_G_loss: %.4f train_D_loss: %.4f || ' % (t_batch+1, epoch, time()-start_time, valid_g_loss[-1], valid_d_loss[-1], g_loss[-1], d_loss[-1])
                    for typ in range(len(g_loss_type)):
                        fmt_str += 'G_%s_loss: %.4f ' % (g_loss_type[typ], valid_g_loss[typ])
                    for typ in range(len(d_loss_type)):
                        fmt_str += 'D_%s_loss: %.4f ' % (d_loss_type[typ], valid_d_loss[typ])
                    print(fmt_str)
                    write_summary(v_input_seq, v_output, v_target, valid_g_loss, valid_d_loss, 'Validating', step)

                    # save models
                    if not np.any(np.isnan(valid_g_loss)):
                        if valid_g_loss[-1] < min_loss:
                            print('\nSaving: lowest G_loss model: best_model')
                            min_loss = valid_g_loss[-1]
                            ckpt_mgr.save()
                        if valid_g_loss[1] < min_berhu:
                            print('\nSaving: lowest Huber loss model: best_berhu')
                            min_berhu = valid_g_loss[1]
                            ckpt_berhu.save()
                        if valid_g_loss[2] < min_ssim:
                            print('\nSaving: lowest SSIM loss model: best_ssim')
                            min_ssim = valid_g_loss[2]
                            ckpt_ssim.save()
                    else:
                        raise ValueError('NaN values appeared')
