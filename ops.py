import tensorflow as tf
import numpy as np
import scipy
import sys
# import cv2
# import network

n_plane = 5  # input planes per sample
n_sample = 16  # samples per stack
# n_target = 30  # number of target planes per input stack
# d_plane = 30  # distance between two adjacent input planes
# dz = 0.2

# random = 0  # randomly sort input planes
# sort = 0  # sort input planes according to dz
crop = 0  # perform random crop to 256*256
normalize = 0  # perform optional complex mean normalization

def mat2npz(f_list: 'list[str]') -> 'list[str]':
    for f in f_list:
        # m = scipy.io.loadmat(f)
        # raw_data = m['rawData']
        # np.savez(f.replace('.mat', '.npz'), raw_data=raw_data)
        m = scipy.io.loadmat(f)
        input_img, target_img = m['inputData'], m['targetData']
        if input_img.ndim == 2:
            input_img = input_img[:,:,np.newaxis]
        if target_img.ndim == 2:
            target_img = target_img[:,:,np.newaxis]
        np.savez(f.replace('.mat', '.npz'), inputData=input_img, targetData=target_img)
    return [f.replace('mat', 'npz') for f in f_list]

def mean_std_norm(x):
    if isinstance(x, tf.Tensor):
        return (x - tf.reduce_mean(x))/tf.reduce_std(x)
    else:
        return (x - x.mean())/x.std()

def min_max_norm(x):
    if isinstance(x, tf.Tensor):
        return (x - tf.reduce_min(x))/(tf.reduce_max(x) - tf.reduce_min(x))
    else:
        return (x - x.min())/(x.max() - x.min())

def complex_mean_norm(x):
    # [H, W, C=2 (real & imaginary)]
    x_comp = (x[:,:,0] + 1j*x[:,:,1])
    x_norm = x_comp / x_comp.mean()
    x_norm = np.stack([np.real(x_norm), np.imag(x_norm)], axis=-1)
    return x_norm

def comp2ap(x):
    if x.ndim == 3:
        tmp = np.zeros_like(x)
        tmp[:,:,0] = np.sqrt(x[:,:,0]**2 + x[:,:,1]**2)
        tmp[:,:,1] = np.arctan2(x[:,:,0], x[:,:,1])
    elif x.ndim == 4:
        tmp = np.zeros_like(x)
        tmp[:,:,:,0] = np.sqrt(x[:,:,:,0]**2 + x[:,:,:,1]**2)
        tmp[:,:,:,1] = np.arctan2(x[:,:,:,0], x[:,:,:,1])
    return tmp

def comp2ap_norm(x):
    if x.ndim == 3:
        tmp = np.zeros_like(x)
        x_norm = (x[:,:,0] + 1j*x[:,:,1])
        x_mean = x_norm.mean()
        x_norm /= x_mean
        tmp[:,:,0] = np.abs(x_norm)
        tmp[:,:,1] = np.angle(x_norm)
    elif x.ndim == 4:
        tmp = np.zeros_like(x)
        x_norm = (x[:,:,:,0] + 1j*x[:,:,:,1])
        x_mean = x_norm.mean(axis=(-2,-1))
        x_norm /= x_mean
        tmp[:,:,:,0] = np.array([np.abs(x) for x in x_norm])
        tmp[:,:,:,1] = np.array([np.angle(x) for x in x_norm])
    return tmp

def total_variation(img, config):
    tv = tf.reduce_mean(tf.image.total_variation(img))
    tv /= config.image_size ** 2
    return tv

def berhu_loss(labels, predictions, delta=0.2, adaptive=True, multiscale=True):
    diff = tf.abs(predictions - labels)
    if adaptive:
        delta = delta * tf.reduce_max(diff)
    loss = tf.reduce_mean(tf.cast(diff <= delta, dtype=tf.float32)*delta*diff + tf.cast(diff > delta, dtype=tf.float32)*(tf.square(diff)/2 + delta**2/2))
    return loss

def load_data(f: 'tf.Tensor') -> '(np.array, np.array)':
    tmp = scipy.io.loadmat(f.numpy())
    assert tmp['inputData'].ndim==3, 'Input data dimension mismatch'
    assert tmp['targetData'].ndim==3, 'Target data dimension mismatch'

    # complex mean normalization (optional), real & imaginary
    input_data = tmp['inputData'].astype('float32')
    target_data = tmp['targetData'].astype('float32')
    if normalize:
        input_tmp = input_data.reshape([input_data.shape[0],input_data.shape[1],input_data.shape[-1]//2,2]).transpose([2,0,1,3])
        input_data = np.array([complex_mean_norm(inp) for inp in input_tmp]).transpose([1,2,0,3]).reshape(input_data.shape)
        target_data = complex_mean_norm(target_data)

    # random crop
    if crop:
        x_min = np.random.randint(low=0, high=input_data.shape[0]-256)
        y_min = np.random.randint(low=0, high=input_data.shape[1]-256)
        input_data = input_data[x_min:x_min+256, y_min:y_min+256]
        target_data = target_data[x_min:x_min+256, y_min:y_min+256]

    inp_idx = np.random.choice(np.arange(input_data.shape[-1]//2), [n_sample, n_plane])
    # [N, T=n_plane, W, H, C=2 (real & imaginary)]
    dataset_input = np.reshape(np.transpose(input_data.reshape([input_data.shape[0],input_data.shape[1],input_data.shape[-1]//2,2])[:,:,inp_idx.reshape(-1),:], [2,0,1,3]),
                                [n_sample, n_plane, input_data.shape[0], input_data.shape[1], 2])
    # [N, W, H, C=2 (real & imaginary)]
    dataset_target = np.repeat([target_data], repeats=n_sample, axis=0)

    print('DEBUG: input shape ', dataset_input.shape)
    return dataset_input, dataset_target

def load_amp_data(f: 'tf.Tensor') -> '(np.array, np.array)':
    tmp = scipy.io.loadmat(f.numpy())
    assert tmp['inputData'].ndim==3, 'Input data dimension mismatch'
    assert tmp['targetData'].ndim==3, 'Target data dimension mismatch'

    # complex mean normalization (optional), real & imaginary
    input_data = tmp['inputData'].astype('float32')
    target_data = tmp['targetData'].astype('float32')

    # random crop
    if crop:
        x_min = np.random.randint(low=0, high=input_data.shape[0]-256)
        y_min = np.random.randint(low=0, high=input_data.shape[1]-256)
        input_data = input_data[x_min:x_min+256, y_min:y_min+256]
        target_data = target_data[x_min:x_min+256, y_min:y_min+256]

    inp_idx = np.random.choice(np.arange(input_data.shape[-1]), [n_sample, n_plane])
    # [N, T=n_plane, W, H, C=2 (real & imaginary)]
    dataset_input = np.reshape(np.transpose(input_data[:,:,inp_idx.reshape(-1),np.newaxis], [2,0,1,3]),
                                [n_sample, n_plane, input_data.shape[0], input_data.shape[1], 1])
    # [N, W, H, C=2 (real & imaginary)]
    dataset_target = np.repeat([target_data], repeats=n_sample, axis=0)
    return dataset_input, dataset_target

def load_test_data(f: 'tf.Tensor') -> '(np.array, np.array)':
    tmp = scipy.io.loadmat(f.numpy())
    assert tmp['inputData'].ndim==3, 'Input data dimension mismatch'
    assert tmp['targetData'].ndim==3, 'Target data dimension mismatch'
    input_data = tmp['inputData'].astype('float32')
    target_data = tmp['targetData'].astype('float32')
    # normalization
    if normalize:
        input_tmp = input_data.reshape([input_data.shape[0],input_data.shape[1],input_data.shape[-1]//2,2]).transpose([2,0,1,3])
        input_data = np.array([complex_mean_norm(inp) for inp in input_tmp]).transpose([1,2,0,3]).reshape(input_data.shape)
        target_data = complex_mean_norm(target_data)

    # [1, T=n_plane, W, H, C=2 (real & imaginary)]
    dataset_input = np.reshape(np.transpose(input_data.reshape([input_data.shape[0],input_data.shape[1],input_data.shape[-1]//2,2]), [2,0,1,3]),
                                [1, input_data.shape[-1]//2, input_data.shape[0], input_data.shape[1], 2])
    # reflect padding
    dataset_input = np.pad(dataset_input, [[0,0],[0,n_plane-dataset_input.shape[1]],[0,0],[0,0],[0,0]], 'reflect')
    dataset_input = np.repeat(dataset_input, np.append(np.ones(input_data.shape[-1]//2-1), n_plane-input_data.shape[-1]//2+1).astype('int'), axis=1)  # replicate padding
    # [1, W, H, C=2 (real & imaginary)]
    dataset_target = target_data[np.newaxis,:,:,:]

    # random crop
    if crop:
        x_min = np.random.randint(low=0, high=dataset_input.shape[0]-256)
        y_min = np.random.randint(low=0, high=dataset_input.shape[1]-256)
        dataset_input = dataset_input[x_min:x_min+256, y_min:y_min+256]
        dataset_target = dataset_target[x_min:x_min+256, y_min:y_min+256]
    return dataset_input, dataset_target

def load_test_amp_data(f: 'tf.Tensor') -> '(np.array, np.array)':
    tmp = scipy.io.loadmat(f.numpy())
    assert tmp['inputData'].ndim==3, 'Input data dimension mismatch'
    assert tmp['targetData'].ndim==3, 'Target data dimension mismatch'

    # complex mean normalization
    input_data = tmp['inputData'].astype('float32')
    target_data = tmp['targetData'].astype('float32')
    # target_norm = (target_data[:,:,0] + 1j*target_data[:,:,1])
    # target_mean = target_norm.mean()
    # input_data /= np.abs(target_mean)
    # target_norm /= target_mean
    # target_data = np.stack([target_norm.real, target_norm.imag], axis=-1)

    # [1, T=n_plane, W, H, C=2 (real & imaginary)]
    input_data = np.pad(input_data[:,:,:,np.newaxis], [[0,0],[0,0],[0,0],[0,1]])
    dataset_input = np.reshape(np.transpose(input_data, [2,0,1,3]),
                                [1, n_plane, input_data.shape[0], input_data.shape[1], 2])
    # [1, W, H, C=2 (real & imaginary)]
    dataset_target = target_data[np.newaxis,:,:,:]
    if resize:
        dataset_input = tf.reshape(tf.image.resize(tf.reshape(dataset_input, [n_plane, input_data.shape[0], input_data.shape[1],1]), [256,256]),
                                [1, n_plane, 256, 256, 1])
        dataset_target = tf.image.resize(dataset_target, [256,256])
    return dataset_input, dataset_target


def dataset_augment(ds):
    ds_f = ds.map(lambda inp, tag: (tf.image.flip_left_right(inp), tf.image.flip_left_right(tag)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.concatenate(ds_f)
    ds_r = ds.map(lambda inp, tag: (tf.image.rot90(inp), tf.image.rot90(tag)), num_parallel_calls=tf.data.experimental.AUTOTUNE)  # rotate 90 degree
    ds = ds.concatenate(ds_r)
    ds_r = ds.map(lambda inp, tag: (tf.image.rot90(inp, k=2), tf.image.rot90(tag, k=2)), num_parallel_calls=tf.data.experimental.AUTOTUNE)  # rotate 180 degree
    return ds.concatenate(ds_r)
