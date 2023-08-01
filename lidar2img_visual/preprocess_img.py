import cv2
from PIL import Image, ImageFilter
import numpy as np

H, W = 900, 1600
fH,fW = 480, 800

def pseudo_telephoto(img, r):
    H, W, C = img.shape
    img = Image.fromarray(np.uint8(img))
    # crop center
    h, w = int(H * r), int(W * r)
    y_offset = (H - h) // 2
    x_offset = (W - w) // 2
    crop = (x_offset, y_offset, x_offset + w, y_offset + h)
    img = img.crop(crop)
    # resize origin
    img = img.resize((W, H))
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return np.array(img).astype(np.float32)


def sample_augmentation():
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int(1 * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    rotate = 0
    return resize, resize_dims, crop, flip, rotate

def img_transform(img, resize_dims, crop):
    img = Image.fromarray(np.uint8(img))
    img = img.resize(resize_dims)
    img = img.crop(crop)
    return np.array(img).astype(np.float32)


if __name__ == "__main__":
    img_front_tele = cv2.imread("img/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg",-1)
    img_front = cv2.imread("img/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg",-1)
    img_front_r = cv2.imread("img/n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT__1531281439770339.jpg",-1)
    img_front_l = cv2.imread("img/n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT__1531281439754844.jpg",-1)
    img_back = cv2.imread("img/n015-2018-07-11-11-54-16+0800__CAM_BACK__1531281439787525.jpg",-1)
    img_back_l = cv2.imread("img/n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT__1531281439797423.jpg",-1)
    img_back_r = cv2.imread("img/n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT__1531281439777893.jpg",-1)

    img_front_tele = pseudo_telephoto(img_front_tele, 0.5)

    _, resize_dims, crop, _, _ = sample_augmentation()

    img_front_tele = img_transform(img_front_tele, resize_dims, crop)
    img_front = img_transform(img_front, resize_dims, crop)
    img_front_r = img_transform(img_front_r, resize_dims, crop)
    img_front_l = img_transform(img_front_l, resize_dims, crop)
    img_back = img_transform(img_back, resize_dims, crop)
    img_back_l = img_transform(img_back_l, resize_dims, crop)
    img_back_r = img_transform(img_back_r, resize_dims, crop)

    cv2.imwrite("n015-2018-07-11-11-54-16+0800__CAM_FRONT_TELE_800x480__1531281439762460.jpg", img_front_tele)
    cv2.imwrite("n015-2018-07-11-11-54-16+0800__CAM_FRONT_800x480__1531281439762460.jpg", img_front)
    cv2.imwrite("n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT_800x480__1531281439770339.jpg", img_front_r)
    cv2.imwrite("n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT_800x480__1531281439754844.jpg", img_front_l)
    cv2.imwrite("n015-2018-07-11-11-54-16+0800__CAM_BACK_800x480__1531281439787525.jpg", img_back)
    cv2.imwrite("n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT_800x480__1531281439797423.jpg", img_back_l)
    cv2.imwrite("n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT_800x480__1531281439777893.jpg", img_back_r)


    

