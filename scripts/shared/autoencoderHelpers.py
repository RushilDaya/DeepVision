import glob
import pickle
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def read_n_images(dataset, start, end, loc):
    """
    Read images (should be jpg) from a dataset (from indexes start to end).
    :param dataset: tuple - (image_name - no file extension, one_hot_label)
    :param start: int - start index
    :param end: int - end index
    :param loc: str - directory location of the images
    :return: numpy, numpy - numpy array of (BGR) images and numpy vector of labels (0 - label_1, 1 - label_2, ...)
    """
    assert glob.glob(loc), "Check directory."
    assert glob.glob("{}/*.jpg".format(loc)), "Check file extension (should be 'jpg')."
    images_list = list(zip(*dataset[start:end]))[0]
    labels_list = list(zip(*dataset[start:end]))[1]
    #labels_list = np.array(labels_list).nonzero()[-1]  # Convert dummy encoding to categorical (one number per category)
    images = [cv2.imread("{}/{}.jpg".format(loc, image)) for image in images_list]
    return np.array(images), labels_list


def generate_img_from_folder(data_dir, mode, batch_size):
    """
    Image generator from directory for fit_generator, eval_generator, predict_generator methods in Keras models. Images must be `.jpg`.
    :param data_dir: str - data directory
    :param mode: str - `training`, `test` or `validate` (must match folder names)
    :param batch_size: int - batch size of generation
    :return: generator - tuples of normalized images as numpy arrays (images/255, images/255)
    """
    loc = "{}/{}".format(data_dir, mode)
    while True:
        with open('{}/labels.pickle'.format(data_dir), 'rb') as f:
            data = pickle.load(f)

        modes = list(data.keys())
        del modes[-1]

        assert mode in modes, "'{}' not a valid mode (must be one of {})".format(mode, str(modes))
        assert glob.glob(loc), "Check directory."
        assert glob.glob("{}/*.jpg".format(loc)), "Check file extension (should be 'jpg')."

        for idx in range(0, len(data[mode]), batch_size):
            start = idx
            end = idx + batch_size

            images, labels = read_n_images(data[mode], start, end, loc)

            yield (images / 255, images / 255)


def get_input_shape(data_dir, mode):
    """
    Get the shape of a (.jpg) image.
    :param data_dir: str - data directory
    :param mode: str - `training`, `test` or `validate` (must match folder names)
    :return: tuple - sample image shape
    """
    loc = "{}/{}".format(data_dir, mode)
    with open('{}/labels.pickle'.format(data_dir), 'rb') as f:
        data = pickle.load(f)

    modes = list(data.keys())
    del modes[-1]
    assert mode in modes, "'{}' not a valid mode (must be one of {})".format(mode, str(modes))
    assert glob.glob(data_dir), "Check directory."
    assert glob.glob("{}/*.jpg".format(loc)), "Check file extension (should be 'jpg')."
    idx = 0  # Arbitrarily chosen
    img = cv2.imread("{}/{}.jpg".format(loc, data[mode][idx][0]))
    return img.shape


def get_num_examples(data_dir, mode):
    """
    Get the number of examples in directory.
    :param data_dir: str - data directory
    :param mode: str - `training`, `test` or `validate` (must match folder names)
    :return: int - number of examples
    """
    with open('{}/labels.pickle'.format(data_dir), 'rb') as f:
        data = pickle.load(f)

    modes = list(data.keys())
    del modes[-1]
    assert mode in modes, "'{}' not a valid mode (must be one of {})".format(mode, str(modes))
    assert glob.glob(data_dir), "Check directory."
    return len(data[mode])


def plot_history(history):
    """
    Plot the training history of Keras model (output of model.fit() or model.fit_generator())
    :param history: tensorflow.python.keras.callbacks.History - Keras object (output from model training)
    :return:
    """
    hist = pd.DataFrame(history.history)
    hist_data = pd.DataFrame(history.history).columns
    hist['epoch'] = history.epoch

    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xlabel('Epoch')
    plt.ylabel(hist_data[1])
    for data in hist_data:
        plt.plot(hist['epoch'], hist[data], label=data)
    plt.title("Epochs vs. Performance")
    plt.grid()
    plt.legend()
    plt.show()


# =============
def get_images(data_dir, mode, idxs):
    """
    Get images from data directory based on index.
    :param data_dir: str - data directory
    :param mode: str - `training`, `test` or `validate` (must match folder names)
    :param idxs: list - list of indexes of images to be retrieved
    :return: numpy - array of images
    """
    loc = "{}/{}".format(data_dir, mode)
    with open('{}/labels.pickle'.format(data_dir), 'rb') as f:
        data = pickle.load(f)

    modes = list(data.keys())
    del modes[-1]
    assert mode in modes, "'{}' not a valid mode (must be one of {})".format(mode, str(modes))
    assert glob.glob(data_dir), "Check directory."
    assert glob.glob("{}/*.jpg".format(loc)), "Check file extension (should be 'jpg')."
    images = []
    for idx in idxs:
        images.append(cv2.imread("{}/{}.jpg".format(loc, data[mode][idx][0])))
    return np.array(images)


def bgr2rgb(img_bgr):
    """
    Conver BGR image (from openCV) to RGB.
    :param img_bgr: numpy - BGR image
    :return: numpy - RGB image
    """
    img_shape = img_bgr.shape
    img_rgb = np.zeros(img_shape)
    img_rgb[:, :, 2] = img_bgr[:, :, 0]
    img_rgb[:, :, 0] = img_bgr[:, :, 2]
    img_rgb[:, :, 1] = img_bgr[:, :, 1]
    return img_rgb.astype('uint8')


def plot_reconstruction(decoded_imgs, data_dir, mode, idxs):
    """
    Plot image reconstruction(s) (original - top and reconstructed - below).
    :param decoded_imgs: numpy - array of reconstruted images
    :param data_dir: str - data directory
    :param mode: str - `training`, `test` or `validate` (must match folder names)
    :param idxs: list - list of indexes of images to be plotted
    :return:
    """
    num_images = len(idxs)
    original_images = get_images(data_dir, mode, idxs)
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(idxs):
        # display original
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(bgr2rgb(original_images[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        decoded_to_plot = decoded_imgs[idx] * 255
        plt.imshow(bgr2rgb(decoded_to_plot))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_encoding(encoded_imgs, data_dir, mode, idxs):
    """
    NOT IMPLEMENTED! Images are not 3 dimensional. Maybe plot each 2D layer.
    :param encoded_imgs: numpy - image encoding
    :param data_dir: data_dir: str - data directory
    :param mode: str - `training`, `test` or `validate` (must match folder names)
    :param idxs: list - list of indexes of images to be plotted
    :return:
    """
    encoded_imgs = encoded_imgs / encoded_imgs.max()  # Normalize the data
    num_images = len(idxs)
    # original_images = get_images(data_dir, mode, idxs) # Original images not used
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(idxs):
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        encoded_to_plot = encoded_imgs[idx] * 255
        plt.imshow(bgr2rgb(encoded_to_plot))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()