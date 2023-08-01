import cv2
import numpy as np

pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# pc_range = [-200, -200, -5.0, 200, 200, 3.0]

img_front_tele = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_FRONT_TELE_800x480__1531281439762460.jpg", -1)
img_front = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_FRONT_800x480__1531281439762460.jpg", -1)
img_front_r = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT_800x480__1531281439770339.jpg", -1)
img_front_l = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT_800x480__1531281439754844.jpg", -1)
img_back = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_BACK_800x480__1531281439787525.jpg", -1)
img_back_l = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT_800x480__1531281439797423.jpg", -1)
img_back_r = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT_800x480__1531281439777893.jpg", -1)

imgs = [img_front_tele, img_front, 
        img_front_r, img_front_l, 
        img_back, 
        img_back_l, img_back_r]

lidar2img = np.fromfile("lidar2imgs_7x4x4.bin",dtype=np.float64)
lidar2img = lidar2img.reshape(7,4,4)
T = lidar2img[0]
print(T)

# H,W,C = imgs[0].shape
H, W = 800, 800
Z = int((pc_range[5]-pc_range[2]) / 0.5)
print("H,W,Z:", H, W, Z)

# normalized grid
xs = np.linspace(0.5, W - 0.5, W, dtype=np.float32) / W
ys = np.linspace(0.5, H - 0.5, H, dtype=np.float32) / H
zs = np.linspace(0.5, Z - 0.5, Z, dtype=np.float32) / Z

# real world grid
xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]
# print("xs:", xs)
# print("ys:", ys)

# homogeneous
v = np.stack(np.meshgrid(xs, ys, zs), axis=-1)
v_h = np.ones((H, W, Z, 4), dtype=np.float32)
v_h[:, :, :, :3] = v[:, :, :, :]

# lidar 2 img
v_h = v_h.reshape((-1, 4))
v_h = v_h.T
ref_points = np.dot(T, v_h)
ref_points = ref_points.T

norm_z = ref_points[:,2]
# norm_z[norm_z < 1e-5] = 1e-5

ref_points[:,0] = ref_points[:,0] / norm_z
ref_points[:,1] = ref_points[:,1] / norm_z

ref_points = ref_points.reshape((H, W, Z, 4))


mask = (ref_points[:, :, :, 2] > 1e-5)
mask = 255 * mask.astype(np.uint8)

z = 8
mapx = ref_points[:, :, z, 0].astype(np.float32)
mapy = ref_points[:, :, z, 1].astype(np.float32)
mask = mask[:, :, z]

# img = cv2.remap(img_front, mapx, mapy, interpolation=cv2.INTER_LINEAR)
# cv2.imwrite("./front_z{}.jpg".format(z), img)
# cv2.imwrite("./front_mask_z{}.jpg".format(z), mask)

img = cv2.remap(img_front_tele, mapx, mapy, interpolation=cv2.INTER_LINEAR)
cv2.imwrite("./front_tele_z{}.jpg".format(z), img)
cv2.imwrite("./front_tele_mask_z{}.jpg".format(z), mask)





