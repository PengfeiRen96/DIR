import numpy as np
import cv2 as cv
import math

IMG_SIZE = 256
HAND_BBOX_RATIO = 0.8
HEATMAP_SIZE = 64
HEATMAP_SIGMA = 2
BLUR_KERNEL = 5
BONE_LENGTH = 0.095


def get_rotation_mat(center, theta=0):
    t = np.deg2rad(theta)
    rotationMat = np.zeros((3, 3), dtype='float32')
    rotationMat[0, 0] = math.cos(t)
    rotationMat[0, 1] = -math.sin(t)
    rotationMat[1, 0] = math.sin(t)
    rotationMat[1, 1] = math.cos(t)
    rotationMat[2, 2] = 1.0
    t = np.matmul((np.identity(3, dtype='float32') - rotationMat), center)
    rotationMat[0, 2] = t[0]
    rotationMat[1, 2] = t[1]
    return rotationMat

def cut_img(img_list, label2d_list, camera=None, radio=0.7, img_size=256, transl=None):
    Min = []
    Max = []
    for label2d in label2d_list:
        Min.append(np.min(label2d, axis=0))
        Max.append(np.max(label2d, axis=0))
    Min = np.min(np.array(Min), axis=0)
    Max = np.max(np.array(Max), axis=0)

    mid = (Min + Max) / 2
    if transl is not None:
        mid = mid + transl
    L = np.max(Max - Min) / 2 / radio
    M = img_size / 2 / L * np.array([[1, 0, L - mid[0]], [0, 1, L - mid[1]]])
    img_list_out = []


    for img in img_list:
        img_list_out.append(cv.warpAffine(img, M, dsize=(img_size, img_size)))

    label2d_list_out = []
    for label2d in label2d_list:
        x = np.concatenate([label2d, np.ones_like(label2d[:, :1])], axis=-1)
        x = x @ M.T
        label2d_list_out.append(x)

    if camera is not None:
        camera[0, 0] = camera[0, 0] * M[0, 0]
        camera[1, 1] = camera[1, 1] * M[1, 1]
        camera[0, 2] = camera[0, 2] * M[0, 0] + M[0, 2]
        camera[1, 2] = camera[1, 2] * M[1, 1] + M[1, 2]

    return img_list_out, label2d_list_out, camera

