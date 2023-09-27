import torch
import random
import pickle
import cv2 as cv
import numpy as np
import os.path as osp
from glob import glob

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from models.manolayer import ManoLayer
import torchvision.transforms as transforms
from utils.utils import get_mano_path, imgUtils


def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1


def fix_obman_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].th_shapedirs[:, 0, :] - mano_layer['right'].th_shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].th_shapedirs[:, 0, :] *= -1


class InterHand_dataset():
    def __init__(self, data_path, split):
        assert split in ['train', 'test', 'val']
        self.split = split

        mano_path = get_mano_path()
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)


        self.data_path = data_path
        self.size = len(glob(osp.join(data_path, split, 'anno', '*.pkl')))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = cv.imread(osp.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))
        mask = cv.imread(osp.join(self.data_path, self.split, 'mask', '{}.jpg'.format(idx)))
        dense = cv.imread(osp.join(self.data_path, self.split, 'dense', '{}.jpg'.format(idx)))

        with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            data = pickle.load(file)

        R = data['camera']['R']
        T = data['camera']['t']
        camera = data['camera']['camera']

        hand_dict = {}
        for hand_type in ['left', 'right']:
            hms = []
            for hIdx in range(7):
                hm = cv.imread(os.path.join(self.data_path, self.split, 'hms', '{}_{}_{}.jpg'.format(idx, hIdx, hand_type)))
                hm = cv.resize(hm, (img.shape[1], img.shape[0]))
                hms.append(hm)

            params = data['mano_params'][hand_type]
            handV, handJ = self.mano_layer[hand_type](torch.from_numpy(params['R']).float(),
                                                      torch.from_numpy(params['pose']).float(),
                                                      torch.from_numpy(params['shape']).float(),
                                                      trans=torch.from_numpy(params['trans']).float())


            handV = handV[0].numpy()
            handJ = handJ[0].numpy()
            handV = handV @ R.T + T
            handJ = handJ @ R.T + T

            handV2d = handV @ camera.T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            handJ2d = handJ @ camera.T
            handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]

            hand_dict[hand_type] = {'hms': hms,
                                    'verts3d': handV, 'joints3d': handJ,
                                    'verts2d': handV2d, 'joints2d': handJ2d,
                                    'R': R @ params['R'][0],
                                    'pose': params['pose'][0],
                                    'shape': params['shape'][0],
                                    'camera': camera
                                    }

        return img, mask, dense, hand_dict


class InterHandDataset(Dataset):
    def __init__(self, data_path, split, img_size=256):
        assert split in ['train', 'test', 'val']
        self.split = split
        mano_path = get_mano_path()
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)

        self.data_path = data_path
        self.imgAug = imgUtils()
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        self.img_size = img_size
        self.size = len(glob(osp.join(data_path, split, 'anno', '*.pkl')))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = cv.imread(osp.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))
        mask = cv.imread(osp.join(self.data_path, self.split, 'mask', '{}.jpg'.format(idx))) # 左右手的语义分割GT
        dense = cv.imread(osp.join(self.data_path, self.split, 'dense', '{}.jpg'.format(idx)))

        with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            data = pickle.load(file)

        R = data['camera']['R']
        T = data['camera']['t']
        camera = data['camera']['camera']

        # Left Hand
        params = data['mano_params']['left']
        handV, handJ = self.mano_layer['left'](torch.from_numpy(params['R']).float(),
                                                      torch.from_numpy(params['pose']).float(),
                                                      torch.from_numpy(params['shape']).float(),
                                                      trans=torch.from_numpy(params['trans']).float())
        handV_left = handV[0].numpy() # 778 * 3
        handJ_left = handJ[0].numpy() # 21 * 3
        handV_left = handV_left @ R.T + T
        handJ_left = handJ_left @ R.T + T

        handV2d_left = handV_left @ camera.T
        handV2d_left_uv = handV2d_left[:, :2] / handV2d_left[:, 2:]
        handJ2d_left = handJ_left @ camera.T
        handJ2d_left_uv = handJ2d_left[:, :2] / handJ2d_left[:, 2:]

        # Right Hand
        params = data['mano_params']['right']
        handV, handJ = self.mano_layer['right'](torch.from_numpy(params['R']).float(),
                                                      torch.from_numpy(params['pose']).float(),
                                                      torch.from_numpy(params['shape']).float(),
                                                      trans=torch.from_numpy(params['trans']).float())
        handV_right = handV[0].numpy()
        handJ_right = handJ[0].numpy()
        handV_right = handV_right @ R.T + T
        handJ_right = handJ_right @ R.T + T

        handV2d_right = handV_right @ camera.T
        handV2d_right_uv = handV2d_right[:, :2] / handV2d_right[:, 2:]
        handJ2d_right = handJ_right @ camera.T
        handJ2d_right_uv = handJ2d_right[:, :2] / handJ2d_right[:, 2:]

        obman_rot_left = data['mano_params']['left']['R'].reshape([3, 3])
        obman_rot_left = np.dot(R, obman_rot_left)
        mano_pose_left = data['mano_params']['left']['pose']
        mano_shape_left = data['mano_params']['left']['shape']

        obman_rot_right = data['mano_params']['right']['R'].reshape([3, 3])
        obman_rot_right = np.dot(R, obman_rot_right)
        mano_pose_right = data['mano_params']['right']['pose']
        mano_shape_right = data['mano_params']['right']['shape']

        do_flip = False
        if self.split == 'train':
            scale, rot, transl_x, transl_y, do_flip = self.imgAug.get_aug_config(0.1, 180, 10, True)

            if do_flip:
                # flip lable
                img_list, label2d_list = self.imgAug.flip([img, mask, dense], [handJ2d_left_uv, handJ2d_right_uv, handV2d_left_uv, handV2d_right_uv], self.img_size)
                handJ2d_right_uv, handJ2d_left_uv, handV2d_right_uv, handV2d_left_uv = label2d_list
                handJ_right, handJ_left, handV_right, handV_left = handJ_left, handJ_right, handV_left, handV_right
                img = img_list[0]
                mask = img_list[1]
                dense = img_list[2]

            rot_aug_mat = np.array([[np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot)), 0],
                                    [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0],
                                    [0, 0, 1]], dtype=np.float32)
            obman_rot_left = np.dot(rot_aug_mat, obman_rot_left)
            obman_rot_right = np.dot(rot_aug_mat, obman_rot_right)

            if random.random() <= 0.3:
                img = self.imgAug.motion_blur(img)

            img_list, label2d_list, label3d_list, _ = self.imgAug.data_augmentation_3D(rot, scale, transl_x, transl_y, camera,
                                                       [img, mask, dense],
                                                       [handJ2d_left_uv, handJ2d_right_uv, handV2d_left_uv, handV2d_right_uv],
                                                       [handJ_left[:, 2:], handJ_right[:, 2:], handV_left[:, 2:], handV_right[:, 2:]],
                                                       img_size=256)
            img = img_list[0]
            mask = img_list[1]
            dense = img_list[2]
            handJ2d_left_uv, handJ2d_right_uv, handV2d_left_uv, handV2d_right_uv = label2d_list
            handJ_left, handJ_right, handV_left, handV_right = label3d_list

        center_left = handJ_left[9:10].copy()
        center_right = handJ_right[9:10].copy()

        seg = np.zeros([self.img_size, self.img_size])
        hand_mask = np.logical_or(mask[:, :, 1] > 50, mask[:, :, 2] > 50)
        hand_mask_left = np.logical_and(hand_mask, mask[:, :, 1] >= mask[:, :, 2])
        hand_mask_right = np.logical_and(hand_mask, mask[:, :, 1] < mask[:, :, 2])
        if do_flip:
            seg[hand_mask_right] = 1
            seg[hand_mask_left] = 2
        else:
            seg[hand_mask_left] = 1
            seg[hand_mask_right] = 2
        seg = seg[np.newaxis, :, :]

        img = self.imgAug.add_noise(img, noise=0.01)
        img = img.astype(np.uint8)
        img_rgb = torch.from_numpy(img).float()
        mask_rgb = torch.from_numpy(mask).float()

        imgTensor = torch.tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255.0
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.normalize_img(imgTensor)
        denseTensor = torch.tensor(dense, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # only normal uv
        handJ2d_left = np.concatenate((handJ2d_left_uv / self.img_size * 2 - 1, handJ_left[:, 2:]), axis=-1)
        handJ2d_right = np.concatenate((handJ2d_right_uv / self.img_size * 2 - 1, handJ_right[:, 2:]) , axis=-1)
        handV2d_left = np.concatenate((handV2d_left_uv / self.img_size * 2 - 1, handV_left[:, 2:]), axis=-1)
        handV2d_right = np.concatenate((handV2d_right_uv / self.img_size * 2 - 1, handV_right[:, 2:]), axis=-1)

        handJ3d_left = handJ_left
        handV3d_left = handV_left
        handJ3d_right = handJ_right
        handV3d_right = handV_right

        inputs = {'img': np.float32(imgTensor), 'img_rgb':np.float32(img_rgb), 'mask_rgb':np.float32(mask_rgb)}
        targets = {'seg': np.float32(seg),
                   'dense': np.float32(denseTensor),
                   'joint_2d_left': np.float32(handJ2d_left),
                   'mesh_2d_left': np.float32(handV2d_left),
                   'joint_2d_right': np.float32(handJ2d_right),
                   'mesh_2d_right': np.float32(handV2d_right),
                   'joint_3d_left': np.float32(handJ3d_left),
                   'mesh_3d_left': np.float32(handV3d_left),
                   'joint_3d_right': np.float32(handJ3d_right),
                   'mesh_3d_right': np.float32(handV3d_right)}
        meta_info = {"camera":  np.float32(camera),
                     "center_left":  np.float32(center_left),
                     "center_right":  np.float32(center_right)}
        return inputs, targets, meta_info

    def uvd2xyz(self, coord, camera):
        fx, fy, fu, fv = camera[:, 0, 0], camera[:, 1, 1], camera[:, 0, 2], camera[:, 1, 2]
        x = (coord[:, :, 0] - fu.unsqueeze(-1)) * coord[:, :, 2] / fx.unsqueeze(-1)
        y = (coord[:, :, 1] - fv.unsqueeze(-1)) * coord[:, :, 2] / fy.unsqueeze(-1)
        coord_xyz = torch.stack((x, y, coord[:, :, 2]), dim=-1)
        return coord_xyz

    def evaluate(self, outs, targets, meta_info):
        device = targets['joint_3d_left'].device
        cube = 1

        joints_left_gt = targets['joint_3d_left'] * cube
        joints_right_gt = targets['joint_3d_right'] * cube
        root_left_gt = joints_left_gt[:, 9:10]
        root_right_gt = joints_right_gt[:, 9:10]
        length_left_gt = torch.linalg.norm(joints_left_gt[:, 9] - joints_left_gt[:, 0], dim=-1)
        length_right_gt = torch.linalg.norm(joints_right_gt[:, 9] - joints_right_gt[:, 0], dim=-1)
        joints_left_gt = joints_left_gt - root_left_gt
        joints_right_gt = joints_right_gt - root_right_gt
        verts_left_gt = targets['mesh_3d_left'] * cube
        verts_right_gt = targets['mesh_3d_right'] * cube

        if outs['pd_joint_xyz_left'] is not None:
            joint_3d_left = outs['pd_joint_xyz_left'].to(device) * cube
            joint_3d_right = outs['pd_joint_xyz_right'].to(device) * cube

            root_left_pred = joint_3d_left[:, 9:10]
            root_right_pred = joint_3d_right[:, 9:10]
            length_left_pred = torch.linalg.norm(joint_3d_left[:, 9] - joint_3d_left[:, 0], dim=-1)
            length_right_pred = torch.linalg.norm(joint_3d_right[:, 9] - joint_3d_right[:, 0], dim=-1)
            scale_left = (length_left_gt / length_left_pred).unsqueeze(-1).unsqueeze(-1)
            scale_right = (length_right_gt / length_right_pred).unsqueeze(-1).unsqueeze(-1)

            joints_left_pred = (joint_3d_left - root_left_pred) * scale_left
            joints_right_pred = (joint_3d_right - root_right_pred) * scale_right

            joint_left_error = torch.linalg.norm((joints_left_pred - joints_left_gt), ord=2, dim=-1)
            joint_left_error = joint_left_error.detach().cpu().numpy().mean() * 1000  # m -> mm

            joint_right_error = torch.linalg.norm((joints_right_pred - joints_right_gt), ord=2, dim=-1)
            joint_right_error = joint_right_error.detach().cpu().numpy().mean() * 1000
        else:
            joint_left_error = 0
            joint_right_error = 0

        if outs['pd_mesh_xyz_left'] is not None:
            verts_left_gt = verts_left_gt - root_left_gt
            verts_right_gt = verts_right_gt - root_right_gt
            mesh_3d_left = outs['pd_mesh_xyz_left'].to(device) * cube
            mesh_3d_right = outs['pd_mesh_xyz_right'].to(device) * cube
            verts_right_pred = (mesh_3d_right - root_right_pred) * scale_right
            verts_left_pred = (mesh_3d_left - root_left_pred) * scale_left
            vert_left_error = torch.linalg.norm((verts_left_pred - verts_left_gt), ord=2, dim=-1)
            vert_left_error = vert_left_error.detach().cpu().numpy().mean() * 1000
            vert_right_error = torch.linalg.norm((verts_right_pred - verts_right_gt), ord=2, dim=-1)
            vert_right_error = vert_right_error.detach().cpu().numpy().mean() * 1000
        else:
            vert_left_error = joint_left_error*0
            vert_right_error = joint_right_error*0

        return joint_left_error, joint_right_error, vert_left_error, vert_right_error
