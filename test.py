from configobj import ConfigObj
import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import scipy.io
import network
import ops
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt

# global parameters
n_channels = 16
n_levels = 5
n_blocks = 5

g_loss_type = ['dis', 'berhu', 'ssim']
d_loss_type = ['fake', 'real']
loss_wt = {'dis': 0.3, 'berhu': 3, 'ssim': 1}

lr_g, lr_d = 1e-5, 1e-6     # learning rate for generator and discriminator respectively

best_model_type = 'best_berhu'     # model, mse, or train

# define the read path and write path
rd_root =   ##### NEED TO FILL IN TESTSET PATH
model_name = 'RH_M_chn=16_mae=3.0_ssim=1.0_lr=1.0e-05-1.0e-06_n=5_trial=mix_backbone'  # the post-fix for tensorboard and model directory
print('model:{}'.format(model_name))
model_dir = os.path.join('Models', model_name)
wt_dir = os.path.join(rd_root, model_name)
if not os.path.exists(model_dir):
    raise FileNotFoundError('Models not found')
if not os.path.exists(wt_dir):
    os.makedirs(wt_dir)


def init_parameters():
    tc = ConfigObj()
    tc.batch_size = 1
    # tc.image_size = 2000  ########### change here
    tc.ker_size = 3
    tc.c_leaky = 0.1
    tc.n_ch = n_channels
    tc.n_blocks = n_blocks
    tc.n_levels = n_levels
    tc.buf_size = 8
    tc.g_loss_type = g_loss_type
    tc.d_loss_type = d_loss_type
    tc.loss_wt = loss_wt
    return tc

def valid_step(input_seq, target_batch):
    # GAN evaluating step
    
    # generator forward inference
    output_batch = gen.inference(input_seq)

    return output_batch

def save_output(input_seq, output_batch, target_batch, full_file):
    output_batch = output_batch.numpy().squeeze()
    target_batch = target_batch.numpy().squeeze()
    input_seq = input_seq.numpy().squeeze()
    # convert to amp/ph
    output_batch = ops.comp2ap_norm(output_batch)
    target_batch = ops.comp2ap_norm(target_batch)
    # save output data
    scipy.io.savemat(full_file + '.mat', {'outputData':output_batch, 'targetData':target_batch})
    # save images
    plt.imsave(full_file+'_output_amp.png', output_batch[:,:,0], vmin=target_batch[:,:,0].min(), vmax=target_batch[:,:,0].max(), cmap=plt.cm.get_cmap('gray'))
    plt.imsave(full_file+'_output_ph.png', output_batch[:,:,1], vmin=-np.pi, vmax=np.pi, cmap=plt.cm.get_cmap('gray'))
    plt.imsave(full_file+'_target_amp.png', target_batch[:,:,0], vmin=target_batch[:,:,0].min(), vmax=target_batch[:,:,0].max(), cmap=plt.cm.get_cmap('gray'))
    plt.imsave(full_file+'_target_ph.png', target_batch[:,:,1], vmin=-np.pi, vmax=np.pi, cmap=plt.cm.get_cmap('gray'))
    for i_seq in range(input_seq.shape[0]):
        plt.imsave(full_file+'_input%d_real.png' % i_seq, input_seq[i_seq,:,:,0], vmin=input_seq[i_seq,:,:,0].min(), vmax=input_seq[i_seq,:,:,0].max(), cmap=plt.cm.get_cmap('gray'))
        plt.imsave(full_file+'_input%d_imag.png' % i_seq, input_seq[i_seq,:,:,1], vmin=input_seq[i_seq,:,:,1].min(), vmax=input_seq[i_seq,:,:,1].max(), cmap=plt.cm.get_cmap('gray'))


if __name__ == '__main__':

    # initialize parameters
    t_config = init_parameters()
    ops.crop = 0

    with tf.device('/cpu:0'):

        test_data = glob.glob(rd_root + '\\*=*.mat')
        print('Total testing = %d' % len(test_data))

        # load the data
        print('Loading data...')
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
        test_dataset = test_dataset.interleave(lambda x:
        tf.data.Dataset.from_tensor_slices(tuple(
            tf.py_function(ops.load_test_data, inp=[x], Tout=(tf.float32, tf.float32)))
        ),
        cycle_length=1, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        test_dataset = test_dataset.batch(t_config.batch_size, drop_remainder=True).prefetch(t_config.buf_size)

    with tf.device('/gpu:0'):

        # create Generator & Discriminator
        gen = network.Generator(t_config)
        dis = network.Discriminator(t_config)

        # create checkpoint
        ckpt = tf.train.Checkpoint(gen=gen, dis=dis)
        ckpt_mgr = tf.train.CheckpointManager(ckpt, directory=model_dir, max_to_keep=2, checkpoint_name=best_model_type)
        if os.path.exists(model_dir + '\\checkpoint'):
            # ckpt.restore(model_dir + '\\best_model-181')
            ckpt.restore(ckpt_mgr.latest_checkpoint)

        # validation
        total_time = 0
        time_list = []
        for v_batch, (v_input_seq, v_target) in test_dataset.enumerate():
            start_time = time()

            print('Debug: input size', v_input_seq.shape)
            print('Debug: input size', v_target.shape)

            # valid step
            v_output = valid_step(input_seq=v_input_seq, target_batch=v_target)

            # timing
            end_time = time()
            time_list.append(end_time - start_time)
            total_time += end_time - start_time
            # save outputs
            save_output(v_input_seq, v_output, v_target, os.path.join(wt_dir, test_data[v_batch].split('\\')[-1].replace('.mat','')))
            np.save(os.path.join(wt_dir, 'time_test.npy'), time_list)
            # report validation time
            print('Validating: iter %d used %.2f sec., average testing time %.2f sec.' % (v_batch.numpy(), end_time-start_time, total_time/(v_batch.numpy()+1)))
        