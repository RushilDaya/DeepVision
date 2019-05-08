import glob
import pickle
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def read_n_images(dataset, start, end, loc):
    assert glob.glob(loc), "Check directory."
    images_list = list(zip(*dataset[start:end]))[0]
    labels_list = list(zip(*dataset[start:end]))[1]
    labels_list = np.array(labels_list).nonzero()[-1]  # Convert dummy encoding to categorical (one number per category)
    images = [cv2.imread("{}/{}.jpg".format(loc, image)) for image in images_list]
    return np.array(images), labels_list


def generate_img_from_folder(data_dir, mode, batch_size, test=False):
    """must be jpg"""
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

            if test:
                yield (images / 255)
            else:
                yield (images / 255, images / 255)  # would be label if not autoencoder


def get_input_shape(data_dir, mode):
    """must be jpg"""
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
    with open('{}/labels.pickle'.format(data_dir), 'rb') as f:
        data = pickle.load(f)

    modes = list(data.keys())
    del modes[-1]
    assert mode in modes, "'{}' not a valid mode (must be one of {})".format(mode, str(modes))
    assert glob.glob(data_dir), "Check directory."
    return len(data[mode])


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist_data = pd.DataFrame(history.history).columns
    hist['epoch'] = history.epoch

    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xlabel('Epoch')
    plt.ylabel(hist_data[1])
    for data in hist_data:
        plt.plot(hist['epoch'], hist[data], label=data)
    # plt.ylim([20,40])
    plt.title("Epochs vs. Performance")
    plt.grid()
    plt.legend()
    plt.show()


# =============
def get_images(data_dir, mode, idxs):
    """must be jpg"""
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
    img_shape = img_bgr.shape
    img_rgb = np.zeros(img_shape)
    img_rgb[:, :, 2] = img_bgr[:, :, 0]
    img_rgb[:, :, 0] = img_bgr[:, :, 2]
    img_rgb[:, :, 1] = img_bgr[:, :, 1]
    return img_rgb.astype('uint8')


def plot_reconstruction(decoded_imgs, data_dir, mode, idxs):
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
    # ===========
    """Haven't tested"""
    # ===========
    encoded_imgs = encoded_imgs / encoded_imgs.max()  # Normalize the data
    num_images = len(idxs)
    original_images = get_images(data_dir, mode, idxs)
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(idxs):
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        encoded_to_plot = encoded_imgs[idx] * 255
        plt.imshow(bgr2rgb(encoded_to_plot))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()