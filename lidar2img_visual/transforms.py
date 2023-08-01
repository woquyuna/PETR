import cv2
import numpy as np

def impad_to_multiple(img, divisor, pad_val=0):
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor

    shape = (pad_h, pad_w, pad_h, pad_w)
    padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    img = cv2.copyMakeBorder(img, padding[1],padding[3],padding[0], padding[2], cv2.BORDER_CONSTANT, value=pad_val)

    return img

def normalize(img, mean, std):
    img = img.astype(np.float32)

    if isinstance(mean, list):
        mean = np.asarray(mean)
    if isinstance(std, list):
        std = np.asarray(std)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))

    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)

    return img.astype(np.float32)
