import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
    Read grayscale image
    Inputs:
    img_path: str: image path
    Returns:
    img: cv2 image
    """
    return cv2.imread(img_path, 0)


import numpy as np


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image
    such that when applying the kernel with the size of filter_size,
    the padded image will be the same size as the original image.

    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter

    Return:
        padded_img: cv2 image: the padding image with replicated edge padding
    """

    height, width = img.shape[:2]
    pad_top = pad_bottom = filter_size // 2
    pad_left = pad_right = filter_size // 2

    padded_img = np.zeros(
        (height + pad_top + pad_bottom, width + pad_left + pad_right), dtype=img.dtype
    )
    top_pad = img[0]
    bottom_pad = img[-1]
    left_pad = np.repeat(img[:, 0], pad_left).reshape(height, pad_left)
    right_pad = np.repeat(img[:, -1], pad_right).reshape(height, pad_right)
    padded_img[:pad_top, pad_left:-pad_right] = top_pad
    padded_img[-pad_bottom:, pad_left:-pad_right] = bottom_pad
    padded_img[pad_top:-pad_bottom, :pad_left] = left_pad
    padded_img[pad_top:-pad_bottom, -pad_right:] = right_pad
    padded_img[:pad_top, :pad_left] = img[0, 0]  # Top-left corner
    padded_img[:pad_top, width + pad_left :] = img[0, -1]  # Top-right corner
    padded_img[height + pad_top :, :pad_left] = img[-1, 0]  # Bottom-left corner
    padded_img[height + pad_top :, width + pad_left :] = img[-1, -1]
    padded_img[pad_top : height + pad_top, pad_left : width + pad_left] = img

    return padded_img


def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size.
    Use replicate padding for the image.
    WARNING: Do not use the exterior functions
    from available libraries such as OpenCV, scikit-image, etc.
    Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    # Need to implement here
    # Pad the image to maintain size after filtering
    padded_img = padding_img(img, filter_size)
    smoothed_img = np.zeros(img.shape, dtype=img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            window_top = y
            window_bottom = window_top + filter_size
            window_left = x
            window_right = window_left + filter_size
            window = padded_img[window_top:window_bottom, window_left:window_right]
            mean_value = np.mean(window, axis=(0, 1))
            smoothed_img[y, x] = mean_value

    return smoothed_img


def median_filter(img, filter_size=3):
    """
    Applies a median filter to an image using replicate padding.

    Args:
        img (np.ndarray): The original image as a NumPy array.
        filter_size (int, optional): The size of the square filter. Defaults to 3.

    Returns:
        np.ndarray: The smoothed image with the median filter.
    """

    padded_img = padding_img(img, filter_size)
    smoothed_img = np.zeros(img.shape, dtype=img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            window_top = y
            window_bottom = window_top + filter_size
            window_left = x
            window_right = window_left + filter_size
            window = padded_img[window_top:window_bottom, window_left:window_right]
            window_flat = window.flatten()
            median_value = np.median(window_flat)
            smoothed_img[y, x] = median_value

    return smoothed_img


def psnr(gt_img, smooth_img):
    """
    Calculates the PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        gt_img (np.ndarray): The ground truth image as a NumPy array.
        smooth_img (np.ndarray): The smoothed image as a NumPy array.

    Returns:
        float: The PSNR score (in dB).
    """
    gt_img = gt_img.astype(np.float64)
    smooth_img = smooth_img.astype(np.float64)
    mse = np.mean((gt_img - smooth_img) ** 2)
    if mse == 0:
        return 100.0  # Arbitrarily set PSNR to 100 for perfect match
    max_pixel = 255.0  # Adjust this if using images with different bit depth
    psnr = 10 * np.log10(max_pixel**2 / mse)

    return psnr


def show_res(before_img, after_img):
    """
    Show the original image and the corresponding smooth image
    Inputs:
        before_img: cv2: image before smoothing
        after_img: cv2: corresponding smoothed image
    Return:
        None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap="gray")
    plt.title("Before")

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap="gray")
    plt.title("After")
    plt.show()


if __name__ == "__main__":
    img_noise = (
        "HW2/ex1_images/noise.png"  # <- need to specify the path to the noise image
    )
    img_gt = "HW2/ex1_images/ori_img.png"  # <- need to specify the path to the gt image
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print("PSNR score of mean filter: ", psnr(img, mean_smoothed_img))
    cv2.imwrite('images_result/mean_filter.png',mean_smoothed_img)
    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print("PSNR score of median filter: ", psnr(img, median_smoothed_img))
    cv2.imwrite("images_result/median_filter.png", median_smoothed_img)
