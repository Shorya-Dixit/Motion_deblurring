import os
from PIL import Image
import numpy as np
import tensorflow as tf


RESHAPE = (256,256)

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def load_image(path):
    img = Image.open(path)
    return img


def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)


def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }

def write_log(callback, names, logs, batch_no):
    """
    Util to write callback for Keras training
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


BASE_DIR = 'weights/'
def save_all_weights(d, g, epoch_number, current_loss):
    g.save_weights(os.path.join(BASE_DIR, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(BASE_DIR, 'discriminator_{}.h5'.format(epoch_number)), True)


#--------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------
#                           For .npy format

def is_an_npy_file(filename):
    # Checks if the given file is a .npy file
    NPY_EXTENSIONS = ['.npy']
    for ext in NPY_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_npy_files(directory):
    # Lists all the .npy files in a directory

    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_npy_file(f)]


def load_npy(path):
    # Loads a .npy file from a given path

    npy = np.load(path)
    return npy


def preprocess_npy(npy):
    # Preprocess the .npy file

    npy = (npy - 127.5) / 127.5
    return npy


import random
def load_npys(path, n_npy):
    # Load n_npy number of .npy files from a given path randomly

    if n_npy < 0:
        n_npy = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_npy_files(A_paths), list_npy_files(B_paths)
    
    zipped_lists = list(zip(all_A_paths, all_B_paths))
    random.shuffle(zipped_lists)

    all_A_paths, all_B_paths = zip(*zipped_lists)

    npy_A, npy_B = [], []
    npy_A_paths, npy_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        np_A, np_B = load_npy(path_A), load_npy(path_B)
        npy_A.append(preprocess_npy(np_A))
        npy_B.append(preprocess_npy(np_B))
        npy_A_paths.append(path_A)
        npy_B_paths.append(path_B)
        if len(npy_A) > n_npy - 1:
            break

    return {
        'A': np.array(npy_A),
        'A_paths': np.array(npy_A_paths),
        'B': np.array(npy_B),
        'B_paths': np.array(npy_B_paths)
    }


import matplotlib.pyplot as plt
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib."""
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    rows = int(np.ceil(n_images / float(cols)))
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, cols, n + 1)
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

