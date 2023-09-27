import numpy as np
import random
import math
import cv2 as cv
import pickle
import torch

import os
import sys
import imgaug.augmenters as iaa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.config import get_cfg_defaults


def projection(scale, trans2d, label3d, img_size=256):
    scale = scale * img_size
    trans2d = trans2d * img_size / 2 + img_size / 2
    trans2d = trans2d

    label2d = scale * label3d[:, :2] + trans2d
    return label2d


# -1, 1 ->
def inv_projection_batch_uv(scale, trans2d, label2d):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label2d: (B x N x 3)
    Returns:
        (B, N, 3)
    """
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d.unsqueeze(1)

    label3d = label2d * scale + trans2d
    return label3d


def projection_batch_xy(scale, trans2d, label3d):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d.unsqueeze(1)

    label2d = scale * label3d[..., :2] + trans2d
    return label2d


# -1, 1 ->
def inv_projection_batch(scale, trans2d, label2d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label2d: (B x N x 3)
    Returns:
        (B, N, 3)
    """
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d.unsqueeze(1)

    label3d = label2d[:, :, :2] * scale + trans2d
    label3d = torch.cat((label3d, label2d[..., 2:]), dim=-1)
    return label3d


def projection_batch(scale, trans2d, label3d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    scale = scale * img_size  # bs
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d * img_size / 2 + img_size / 2  # bs x 2
    trans2d = trans2d.unsqueeze(1)

    label2d = scale * label3d[..., :2] + trans2d
    return label2d


def projection_batch_np(scale, trans2d, label3d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    scale = scale * img_size  # bs
    if scale.dim() == 1:
        scale = scale[..., np.newaxis, np.newaxis]
    if scale.dim() == 2:
        scale = scale[..., np.newaxis]
    trans2d = trans2d * img_size / 2 + img_size / 2  # bs x 2
    trans2d = trans2d[:, np.newaxis, :]

    label2d = scale * label3d[..., :2] + trans2d
    return label2d

def uvd2world(uvd, camera, R, T, rot=None):
    if rot is not None:
        device = uvd.device
        uv_pad = torch.cat((uvd[..., :2], torch.ones_like(uvd[..., :1]).to(device)), dim=-1)
        uv = torch.matmul(uv_pad, torch.inverse(torch.transpose(rot, -1, -2)))
        uvd = torch.cat((uv[..., :2], uvd[..., 2:]), dim=-1)
    xyz = uvd2xyz(uvd, camera)
    xyz_world = xyz - T
    xyz_world = torch.matmul(xyz_world, R)
    return xyz_world

def nuvd2world(nuvd, camera, R, T, rot=None, img_size=256):
    uv = (nuvd[..., :2] + 1) / 2 * img_size
    uvd = torch.cat((uv, nuvd[..., 2:]), dim=-1)
    xyz_world = uvd2world(uvd, camera, R, T, rot)
    return xyz_world


def world2uvd(xyz_world, camera, R, T, rot=None):
    xyz_cam = torch.matmul(xyz_world, torch.transpose(R, -1, -2)) + T
    uvd = xyz2uvd(xyz_cam, camera)
    if rot is not None:
        device = uvd.device
        uv_pad = torch.cat((uvd[..., :2], torch.ones_like(uvd[..., :1]).to(device)), dim=-1)
        uv = torch.matmul(uv_pad, torch.transpose(rot, -1, -2))
        uvd = torch.cat((uv[..., :2], uvd[..., 2:]), dim=-1)
    return uvd


def world2nuvd(xyz, camera, R, T, rot=None, img_size=256):
    uvd = world2uvd(xyz, camera, R, T, rot)
    uv = uvd[..., :2] / img_size * 2 - 1
    uvd = torch.cat((uv, uvd[..., 2:]), dim=-1)
    return uvd


def uvd2xyz(uvd, camera):
    fx, fy, fu, fv = camera[..., 0:1, 0:1], camera[..., 1:2, 1:2], camera[..., 0:1, 2:3], camera[..., 1:2, 2:3]
    x = (uvd[..., 0:1] - fu) * uvd[..., 2:3] / fx
    y = (uvd[..., 1:2] - fv) * uvd[..., 2:3] / fy
    xyz = torch.cat((x, y, uvd[..., 2:3]), dim=-1)
    return xyz


# mm -> img_size
def xyz2uv(xyz, camera):
    fx, fy, fu, fv = camera[..., 0:1, 0:1], camera[..., 1:2, 1:2], camera[..., 0:1, 2:3], camera[..., 1:2, 2:3]
    u = (xyz[..., 0:1] * fx / (xyz[..., 2:3] + 1e-8) + fu)
    v = (xyz[..., 1:2] * fy / (xyz[..., 2:3] + 1e-8) + fv)
    return torch.cat((u, v), dim=-1)

def xyz2uvd(xyz, camera):
    fx, fy, fu, fv = camera[..., 0:1, 0:1], camera[..., 1:2, 1:2], camera[..., 0:1, 2:3], camera[..., 1:2, 2:3]
    u = (xyz[..., 0:1] * fx / (xyz[..., 2:3] + 1e-8) + fu)
    v = (xyz[..., 1:2] * fy / (xyz[..., 2:3] + 1e-8) + fv)
    uvd = torch.cat((u, v, xyz[..., 2:3]), dim=-1)
    return uvd

# mm -> [-1,1]
def xyz2nuv(xyz, camera, img_size=256):
    fx, fy, fu, fv = camera[..., 0:1, 0:1], camera[..., 1:2, 1:2], camera[..., 0:1, 2:3], camera[..., 1:2, 2:3]
    u = (xyz[..., 0:1] * fx / (xyz[..., 2:3]+1e-8) + fu)
    v = (xyz[..., 1:2] * fy / (xyz[..., 2:3]+1e-8) + fv)
    uv = torch.cat((u, v), dim=-1) / img_size * 2 - 1
    return uv

def xyz2nuvd(xyz, camera, img_size=256):
    fx, fy, fu, fv = camera[..., 0:1, 0:1], camera[..., 1:2, 1:2], camera[..., 0:1, 2:3], camera[..., 1:2, 2:3]
    u = (xyz[..., 0:1] * fx / (xyz[..., 2:3]+1e-8) + fu)
    v = (xyz[..., 1:2] * fy / (xyz[..., 2:3]+1e-8) + fv)
    uv = torch.cat((u, v), dim=-1) / img_size * 2 - 1
    uvd = torch.cat((uv, xyz[..., 2:3]), dim=-1)
    return uvd

def uvd2xyz_np(coord, camera):
    fx, fy, fu, fv = camera[..., 0:1, 0:1], camera[..., 1:2, 1:2], camera[..., 0:1, 2:3], camera[..., 1:2, 2:3]
    x = (coord[..., 0:1] - fu) * coord[..., 2:3] / fx
    y = (coord[..., 1:2] - fv) * coord[..., 2:3] / fy
    coord_xyz = np.concatenate((x, y, coord[..., 2:]), axis=-1)
    return coord_xyz
def xyz2uvd_np(xyz, camera):
    fx, fy, fu, fv = camera[..., 0:1, 0:1], camera[..., 1:2, 1:2], camera[..., 0:1, 2:3], camera[..., 1:2, 2:3]
    u = (xyz[..., 0:1] * fx / (xyz[..., 2:3] + 1e-8) + fu)
    v = (xyz[..., 1:2] * fy / (xyz[..., 2:3] + 1e-8) + fv)
    uvd = np.concatenate((u, v, xyz[..., 2:3]), axis=-1)
    return uvd
def get_mano_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(abspath, cfg.MISC.MANO_PATH)
    mano_path = {'left': os.path.join(path, 'MANO_LEFT.pkl'),
                 'right': os.path.join(path, 'MANO_RIGHT.pkl')}
    return mano_path


def get_graph_dict_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    graph_path = {'left': os.path.join(abspath, cfg.MISC.GRAPH_LEFT_DICT_PATH),
                  'right': os.path.join(abspath, cfg.MISC.GRAPH_RIGHT_DICT_PATH)}
    return graph_path


def get_dense_color_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dense_path = os.path.join(abspath, cfg.MISC.DENSE_COLOR)
    return dense_path


def get_mano_seg_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    seg_path = os.path.join(abspath, cfg.MISC.MANO_SEG_PATH)
    return seg_path


def get_upsample_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    upsample_path = os.path.join(abspath, cfg.MISC.UPSAMPLE_PATH)
    return upsample_path





class imgUtils():
    def __init__(self, coco_path=''):
        super(imgUtils, self).__init__()
        self.seq = iaa.Sequential([
            iaa.Sometimes(0.3, iaa.MotionBlur(k=(3, 15), order=0))
        ])
        if coco_path != '':
            self.coco_path = coco_path
            file = open(self.coco_path + '/images.txt', 'r')
            self.bk_list = file.readlines()
            self.bk_num = len(self.bk_list)

        # self.motion_blur = kau.RandomMotionBlur((3, 7), 180, 0.5, p=1., keepdim=True)


    @ staticmethod
    def pad2squre(img, color=None):
        if img.shape[0] > img.shape[1]:
            W = img.shape[0] - img.shape[1]
        else:
            W = img.shape[1] - img.shape[0]
        W1 = int(W / 2)
        W2 = W - W1
        if color is None:
            if img.shape[2] == 3:
                color = (0, 0, 0)
            else:
                color = 0
        if img.shape[0] > img.shape[1]:
            return cv.copyMakeBorder(img, 0, 0, W1, W2, cv.BORDER_CONSTANT, value=color)
        else:
            return cv.copyMakeBorder(img, W1, W2, 0, 0, cv.BORDER_CONSTANT, value=color)

    @ staticmethod
    def cut2squre(img):
        if img.shape[0] > img.shape[1]:
            s = int((img.shape[0] - img.shape[1]) / 2)
            return img[s:(s + img.shape[1])]
        else:
            s = int((img.shape[1] - img.shape[0]) / 2)
            return img[:, s:(s + img.shape[0])]

    @ staticmethod
    def get_scale_mat(center, scale=1.0):
        scaleMat = np.zeros((3, 3), dtype='float32')
        scaleMat[0, 0] = scale
        scaleMat[1, 1] = scale
        scaleMat[2, 2] = 1.0
        t = np.matmul((np.identity(3, dtype='float32') - scaleMat), center)
        scaleMat[0, 2] = t[0]
        scaleMat[1, 2] = t[1]
        return scaleMat

    @ staticmethod
    def get_rotation_mat(center, theta=0):
        # t = theta * (3.14159 / 180)
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

    @ staticmethod
    def get_rotation_mat3d(theta=0):
        # t = theta * (3.14159 / 180)
        t = np.deg2rad(theta)
        rotationMat = np.zeros((3, 3), dtype='float32')
        rotationMat[0, 0] = math.cos(t)
        rotationMat[0, 1] = -math.sin(t)
        rotationMat[1, 0] = math.sin(t)
        rotationMat[1, 1] = math.cos(t)
        rotationMat[2, 2] = 1.0
        return rotationMat

    @ staticmethod
    def get_affine_mat(theta=0, scale=1.0,
                       u=0, v=0,
                       height=480, width=640):
        center = np.array([width / 2, height / 2, 1], dtype='float32')
        rotationMat = imgUtils.get_rotation_mat(center, theta)
        scaleMat = imgUtils.get_scale_mat(center, scale)
        trans = np.identity(3, dtype='float32')
        trans[0, 2] = u
        trans[1, 2] = v
        affineMat = np.matmul(scaleMat, rotationMat)
        affineMat = np.matmul(trans, affineMat)
        return affineMat

    @staticmethod
    def img_trans(theta, scale, u, v, img):
        size = img.shape[0]
        u = int(u * size / 2)
        v = int(v * size / 2)
        affineMat = imgUtils.get_affine_mat(theta=theta, scale=scale,
                                            u=u, v=v,
                                            height=256, width=256)
        return cv.warpAffine(src=img,
                             M=affineMat[0:2, :],
                             dsize=(256, 256),
                             dst=img,
                             flags=cv.INTER_LINEAR,
                             borderMode=cv.BORDER_REPLICATE,
                             borderValue=(0, 0, 0)
                             )

    @staticmethod
    def data_augmentation(theta, scale, u, v,
                          img_list=None, label2d_list=None, label3d_list=None,
                          R=None,
                          img_size=224):
        affineMat = imgUtils.get_affine_mat(theta=theta, scale=scale,
                                            u=u, v=v,
                                            height=img_size, width=img_size)
        if img_list is not None:
            img_list_out = []
            for img in img_list:
                img_list_out.append(cv.warpAffine(src=img,
                                                  M=affineMat[0:2, :],
                                                  dsize=(img_size, img_size)))
        else:
            img_list_out = None

        if label2d_list is not None:
            label2d_list_out = []
            for label2d in label2d_list:
                label2d_list_out.append(np.matmul(label2d, affineMat[0:2, 0:2].T) + affineMat[0:2, 2:3].T)
        else:
            label2d_list_out = None

        if label3d_list is not None:
            label3d_list_out = []
            R_delta = imgUtils.get_rotation_mat3d(theta)
            for label3d in label3d_list:
                label3d_list_out.append(np.matmul(label3d, R_delta.T))
        else:
            label3d_list_out = None

        if R is not None:
            R_delta = imgUtils.get_rotation_mat3d(theta)
            R = np.matmul(R_delta, R)
        else:
            R = imgUtils.get_rotation_mat3d(theta)

        return img_list_out, label2d_list_out, label3d_list_out, R
    @staticmethod
    def data_augmentation_3D(theta, scale, u, v, cam,
                          img_list=None, label2d_list=None, depth_list=None,
                          R=None,
                          img_size=224):
        affineMat = imgUtils.get_affine_mat(theta=theta, scale=scale,
                                            u=u, v=v,
                                            height=img_size, width=img_size)
        if img_list is not None:
            img_list_out = []
            for img in img_list:
                img_list_out.append(cv.warpAffine(src=img,
                                                  M=affineMat[0:2, :],
                                                  dsize=(img_size, img_size)))
        else:
            img_list_out = None

        if label2d_list is not None and depth_list is not None:
            label2d_list_out = []
            label3d_list_out = []
            for i in range(len(label2d_list)):
                label2d, depth = label2d_list[i], depth_list[i]
                label2d_aug = np.matmul(label2d, affineMat[0:2, 0:2].T) + affineMat[0:2, 2:3].T
                label2d_list_out.append(label2d_aug)
                labeluvd_aug = np.concatenate((label2d_aug, depth), axis=-1)
                label3d_list_out.append(uvd2xyz_np(labeluvd_aug, cam))
        else:
            label2d_list_out = None
            label3d_list_out = None

        if R is not None:
            R_delta = imgUtils.get_rotation_mat3d(theta)
            R = np.matmul(R_delta, R)
        else:
            R = imgUtils.get_rotation_mat3d(theta)

        return img_list_out, label2d_list_out, label3d_list_out, R



    @ staticmethod
    def add_noise(img, noise=0.00, scale=255.0, alpha=0.3, beta=0.05):
        # add brightness noise & add random gaussian noise
        a = np.random.uniform(1 - alpha, 1 + alpha, 3)
        b = scale * beta * (2 * random.random() - 1)
        img = a * img + b + scale * np.random.normal(loc=0.0, scale=noise, size=img.shape)
        img = np.clip(img, 0, scale).astype(np.uint8)
        return img

    @ staticmethod
    def aug_color(img, color_factor=0.2):
        c_up = 1.0 + color_factor
        c_low = 1.0 - color_factor
        color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
        img = np.clip(img * color_scale[None, None, :], 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def get_aug_config(scale_factor=0.1, rot_factor=180, transl_factor=10, flip=True):
        scale = 1 + (np.random.rand() * 2 - 1) * scale_factor
        rot = (np.random.rand() * 2 - 1) * rot_factor
        transl_x = (np.random.rand() * 2 - 1) * transl_factor
        transl_y = (np.random.rand() * 2 - 1) * transl_factor
        if flip:
            do_flip = random.random() <= 0.5
        else:
            do_flip = False

        return scale, rot, transl_x, transl_y, do_flip

    @staticmethod
    def flip(img_list=None, label2d_list=None, img_size=256):
        if img_list is not None:
            img_list_out = []
            for img in img_list:
                img_list_out.append(img[:, ::-1, :].copy())
        else:
            img_list_out = None

        if label2d_list is not None:
            label2d_list_out = []
            for label2d in label2d_list:
                label2d_out = label2d.copy()
                label2d_out[:, 0:1] = img_size - label2d_out[:, 0:1] - 1
                label2d_list_out.append(label2d_out)
        else:
            label2d_list_out = None

        return img_list_out, label2d_list_out

    @staticmethod
    def bi_flip(img_list=None, label2d_list=None, img_size=256):
        flip_direction = 0
        if random.random() <= 0.5:
            flip_direction = 1

        if img_list is not None:
            img_list_out = []
            for img in img_list:
                if flip_direction == 0:
                    img_list_out.append(img[:, ::-1, :])
                else:
                    img_list_out.append(img[::-1, :, :])
        else:
            img_list_out = None

        if label2d_list is not None:
            label2d_list_out = []
            for label2d in label2d_list:
                label2d_out = label2d.copy()
                if flip_direction == 0:
                    label2d_out[:, 0:1] = img_size - label2d_out[:, 0:1] - 1
                else:
                    label2d_out[:, 1:2] = img_size - label2d_out[:, 1:2] - 1
                label2d_list_out.append(label2d_out)
        else:
            label2d_list_out = None

        return img_list_out, label2d_list_out

    @staticmethod
    def motion_blur(image, max_size=10):
        size = np.random.randint(3, max_size)
        angle = np.random.uniform(-180, 180) * np.pi / 180
        k = np.zeros((size, size), dtype=np.float32)
        k[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
        k = cv.warpAffine(k, cv.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
        k = k * (1.0 / np.sum(k))
        return cv.filter2D(image, -1, k)

    def blur(self, img):
        return self.seq(image=img)

    def tensor_rand_bk(self, batch_size):
        rand_id = np.random.randint(0, self.bk_num, [batch_size])
        bk_list = []
        for index in rand_id:
            bk = cv.imread(self.coco_path+'/' + self.bk_list[index].rstrip("\n"))
            H, W, C = bk.shape
            center_y, center_x = H//2, W//2
            bbox_len = min(H//2, W//2)
            x0, y0 = center_x - bbox_len, center_y - bbox_len
            x1, y1 = center_x + bbox_len, center_y + bbox_len
            bk_crop = self.cut_img(bk, (x0, x1, y0, y1))
            bk_crop = cv.resize(bk_crop, (self.image_size, self.image_size))
            bk_list.append(bk_crop)
        bks = np.stack(bk_list, axis=0)[..., ::-1] / 255
        return torch.from_numpy(bks.copy()).float().cuda().permute(0, 3, 1, 2)

    def cut_img(self, img, bbox):
        cut = img[max(int(bbox[2]), 0):min(int(bbox[3]), img.shape[0]),
              max(int(bbox[0]), 0):min(int(bbox[1]), img.shape[1])]
        cut = cv.copyMakeBorder(cut,
                                max(int(-bbox[2]), 0),
                                max(int(bbox[3] - img.shape[0]), 0),
                                max(int(-bbox[0]), 0),
                                max(int(bbox[1] - img.shape[1]), 0),
                                borderType=cv2.BORDER_CONSTANT,
                                value=(0, 0, 0))
        return cut

    def tensor_motion_blur(self, img_mask, tex):
        img = img_mask[:, :3, ...]
        mask = img_mask[:, 3:4, ...]
        B = img.size(0)
        tex_mean = (img.reshape(B, 3, -1)*mask.reshape(B, 1, -1)).mean(-1) / ((mask.reshape(B, 1, -1)).mean(-1) + 1e-8)
        tex_mean = tex_mean.reshape([B, 3, 1, 1])
        combine_img = img*mask + tex_mean* torch.ones_like(img).cuda()*(1-mask)
        img_mask = torch.cat((combine_img, mask), dim=1)
        out = self.motion_blur(img_mask.contiguous())
        return out[:, :3, ...], out[:, 3:, ...].gt(0).float()

