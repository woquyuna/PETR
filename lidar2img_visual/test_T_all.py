import cv2
import numpy as np

pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

img_f = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg")
img_fr = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT__1531281439770339.jpg")
img_fl = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT__1531281439754844.jpg")
img_b = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_BACK__1531281439787525.jpg")
img_bl = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT__1531281439797423.jpg")
img_br = cv2.imread("n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT__1531281439777893.jpg")

# img group
imgs = [img_f, img_fr, img_fl, img_b, img_bl, img_br]
# lidar2img
lidar2img = np.fromfile("lidar2img_0.npy",dtype=np.float32)

H, W, C = imgs[0].shape
Z = int(pc_range[5]-pc_range[2])
print("H,W,Z:", H, W, Z)

# build voxel
# normalized grid
xs = np.linspace(0.5, W - 0.5, W, dtype=np.float32) / W
ys = np.linspace(0.5, H - 0.5, H, dtype=np.float32) / H
zs = np.linspace(0.5, Z - 0.5, Z, dtype=np.float32) / Z

# lidar grid
xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]

v = np.stack(np.meshgrid(xs, ys, zs), axis=-1)  # voxel
vh = np.ones((H, W, Z, 4), dtype=np.float32)   # homogeneous voxel
vh[:, :, :, :3] = v[:, :, :, :]


outputs = []
output_masks = []
z = 0
for i in range(len(imgs)):
    img = imgs[i]
    T = lidar2img[0][i]

    # voxel to img
    # flatten to transform
    vh_ = vh.copy()
    vh_ = vh_.reshape((-1, 4))
    vh_ = vh_.T
    ref_points = np.dot(T, vh_)
    ref_points = ref_points.T

    # normalize with z
    ref_points[:, 0] = ref_points[:, 0] / (ref_points[:, 2]+1e-5)
    ref_points[:, 1] = ref_points[:, 1] / (ref_points[:, 2]+1e-5)

    # back to voxel
    ref_points = ref_points.reshape((H, W, Z, 4))

    # generate mask: z <= 0 is not valid
    mask = (ref_points[:, :, :, 2] > 1e-5)
    mask = mask.astype(np.float32)
    # mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)

    # assume z = 0
    # z = 0
    mapx = ref_points[:, :, z, 0].astype(np.float32)
    mapy = ref_points[:, :, z, 1].astype(np.float32)
    mask = mask[:, :, z]

    # warp img
    img_warp = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
    img_warp = img_warp.astype(np.float32) * mask

    outputs.append(img_warp)
    output_masks.append(mask)

    # save img
    # cv2.imwrite("./img{}_z{}.jpg".format(i,z), img_warp.astype(np.uint8))
    mask = mask * 255
    # cv2.imwrite("./img{}_z{}_validz.jpg".format(i,0), mask.astype(np.uint8))

# weighted merge img
img_out = outputs[0]
mask_out = (np.mean(img_out, axis=-1) > 0).astype(np.float32)
for i in range(1,len(outputs)):
    img_out = img_out + outputs[i]
    mask = (np.mean(outputs[i], axis=-1) > 0).astype(np.float32)
    mask_out = mask_out + mask

mask_viz = mask_out / 6 * 255
# cv2.imwrite("./img_merge_mask_z{}.jpg".format(z), mask_viz.astype(np.uint8))

mask_out = np.repeat(np.expand_dims(mask_out, axis=-1), 3, axis=-1)

img_out = img_out / (mask_out + 1e-5)
img_out = img_out.astype(np.uint8)
img_out = cv2.flip(img_out, 0)

cv2.imwrite("./img_merge_z{}.jpg".format(z), img_out)




