import argparse
import cv2 as cv
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.manolayer import ManoLayer
from utils.utils import get_mano_path
from dataset.dataset_utils import IMG_SIZE
from dataset.interhand import fix_shape, InterHand_dataset
from models.dir import DIR


class Jr():
    def __init__(self, J_regressor, device='cuda'):
        self.device = device
        self.process_J_regressor(J_regressor)

    def process_J_regressor(self, J_regressor):
        J_regressor = J_regressor.clone().detach()
        tip_regressor = torch.zeros_like(J_regressor[:5])
        tip_regressor[0, 745] = 1.0
        tip_regressor[1, 317] = 1.0
        tip_regressor[2, 444] = 1.0
        tip_regressor[3, 556] = 1.0
        tip_regressor[4, 673] = 1.0
        J_regressor = torch.cat([J_regressor, tip_regressor], dim=0)
        new_order = [0, 13, 14, 15, 16,
                     1, 2, 3, 17,
                     4, 5, 6, 18,
                     10, 11, 12, 19,
                     7, 8, 9, 20]
        self.J_regressor = J_regressor[new_order].contiguous().to(self.device)

    def __call__(self, v):
        return torch.matmul(self.J_regressor, v)


class handDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask, dense, hand_dict = self.dataset[idx]
        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        imgTensor = torch.tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.normalize_img(imgTensor)

        maskTensor = torch.tensor(mask, dtype=torch.float32) / 255

        joints_left_gt = torch.from_numpy(hand_dict['left']['joints3d']).float()
        joints_right_gt = torch.from_numpy(hand_dict['right']['joints3d']).float()
        verts_left_gt = torch.from_numpy(hand_dict['left']['verts3d']).float()
        verts_right_gt = torch.from_numpy(hand_dict['right']['verts3d']).float()

        joints_2d_left_gt = torch.from_numpy(hand_dict['left']['joints2d']).float()
        joints_2d_right_gt = torch.from_numpy(hand_dict['right']['joints2d']).float()
        verts_2d_left_gt = torch.from_numpy(hand_dict['left']['verts2d']).float()
        verts_2d_right_gt = torch.from_numpy(hand_dict['right']['verts2d']).float()

        cam = torch.from_numpy(hand_dict['left']['camera']).float()
        return imgTensor, maskTensor, \
            joints_left_gt, verts_left_gt, joints_right_gt, verts_right_gt, \
            joints_2d_left_gt, verts_2d_left_gt, joints_2d_right_gt, verts_2d_right_gt, cam


def xyz2uvd(joint, cam):
    joint_2d = torch.matmul(joint, cam.permute(0, 2, 1))
    joint_2d = joint_2d[:, :, :2] / joint_2d[:, :, 2:]
    return joint_2d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default='./DIR.pth')
    parser.add_argument("--data_path", type=str, default='./data/interhand2.6m/')
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--root_joint", type=int, default=0) # 0 Wrist 9 MCP
    parser.add_argument("--scale", type=bool, default=True)

    opt = parser.parse_args()
    opt.map = False
    name = 'DIR-PoseEmb-Wrist'

    file_folder = './result/%s' % name
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)

    network = DIR(21, './misc/mano').cuda()
    stage_num = 3

    state = torch.load(opt.model, map_location='cpu')['net']
    network.load_state_dict(state, strict=False)

    network.eval()
    network.cuda()

    mano_path = get_mano_path()
    mano_layer = {'left': ManoLayer(mano_path['left'], center_idx=None),
                  'right': ManoLayer(mano_path['right'], center_idx=None)}
    fix_shape(mano_layer)
    J_regressor = {'left': Jr(mano_layer['left'].J_regressor),
                   'right': Jr(mano_layer['right'].J_regressor)}

    faces_left = mano_layer['left'].get_faces()
    faces_right = mano_layer['right'].get_faces()

    dataset = handDataset(InterHand_dataset(opt.data_path, split='test'))
    dataloader = DataLoader(dataset, batch_size=opt.bs, shuffle=False,
                            num_workers=8, drop_last=False, pin_memory=True)

    joints_loss = {'left': [], 'right': []}
    verts_loss = {'left': [], 'right': []}
    joints_xyz_list = {'left': [], 'right': []}
    joints_xyz_gt_list = {'left': [], 'right': []}
    joints_2d_loss = {'left': [], 'right': []}
    verts_2d_loss = {'left': [], 'right': []}
    root_loss_list = []
    inter_volume_list = []
    val_root_list = []
    idx = 0

    with torch.no_grad():
        for data in tqdm(dataloader):
            imgTensors = data[0].cuda()
            joints_left_gt = data[2].cuda()
            verts_left_gt = data[3].cuda()
            joints_right_gt = data[4].cuda()
            verts_right_gt = data[5].cuda()
            joints_2d_left_gt = data[6].cuda()
            verts_2d_left_gt = data[7].cuda()
            joints_2d_right_gt = data[8].cuda()
            verts_2d_right_gt = data[9].cuda()
            cam = data[10].cuda()

            joints_left_gt = J_regressor['left'](verts_left_gt)
            joints_right_gt = J_regressor['right'](verts_right_gt)
            joints_2d_left_gt = xyz2uvd(joints_left_gt, cam)
            joints_2d_right_gt = xyz2uvd(joints_right_gt, cam)

            gt_offset = joints_right_gt[:, opt.root_joint:opt.root_joint+1] - joints_left_gt[:, opt.root_joint:opt.root_joint+1]
            root_left_gt = joints_left_gt[:, opt.root_joint:opt.root_joint+1].clone()
            root_right_gt = joints_right_gt[:, opt.root_joint:opt.root_joint+1].clone()

            length_left_gt = torch.linalg.norm(joints_left_gt[:, 9] - joints_left_gt[:, 0], dim=-1)
            length_right_gt = torch.linalg.norm(joints_right_gt[:, 9] - joints_right_gt[:, 0], dim=-1)
            joints_left_gt = joints_left_gt - root_left_gt
            verts_left_gt = verts_left_gt - root_left_gt
            joints_right_gt = joints_right_gt - root_right_gt
            verts_right_gt = verts_right_gt - root_right_gt

            input_list = {'img': imgTensors}
            result, _ = network(input_list, None, None)

            rel_root_pred = result[stage_num - 1]['pd_offset'].unsqueeze(1) * 0.15
            verts_left_pred = result[stage_num - 1]['pd_mesh_xyz_left']
            verts_right_pred = result[stage_num - 1]['pd_mesh_xyz_right']
            joints_left_pred_ori = J_regressor['left'](verts_left_pred)
            joints_right_pred_ori = J_regressor['right'](verts_right_pred)

            root_left_pred = joints_left_pred_ori[:, opt.root_joint:opt.root_joint+1].clone()
            root_right_pred = joints_right_pred_ori[:, opt.root_joint:opt.root_joint+1].clone()
            length_left_pred = torch.linalg.norm(joints_left_pred_ori[:, 9] - joints_left_pred_ori[:, 0], dim=-1)
            length_right_pred = torch.linalg.norm(joints_right_pred_ori[:, 9] - joints_right_pred_ori[:, 0], dim=-1)
            scale_left = (length_left_gt / length_left_pred).unsqueeze(-1).unsqueeze(-1)
            scale_right = (length_right_gt / length_right_pred).unsqueeze(-1).unsqueeze(-1)

            if not opt.scale:
                scale_left = 1
                scale_right = 1

            joints_left_pred = (joints_left_pred_ori - root_left_pred) * scale_left
            verts_left_pred = (verts_left_pred - root_left_pred) * scale_left
            joints_right_pred = (joints_right_pred_ori - root_right_pred) * scale_right
            verts_right_pred = (verts_right_pred - root_right_pred) * scale_right

            joint_left_loss = torch.linalg.norm((joints_left_pred - joints_left_gt), ord=2, dim=-1)
            joint_left_loss = joint_left_loss.detach().cpu().numpy()
            joints_loss['left'].append(joint_left_loss)
            joints_xyz_list['left'].append(joints_left_pred.detach().cpu().numpy())
            joints_xyz_gt_list['left'].append(joints_left_gt.detach().cpu().numpy())

            joint_right_loss = torch.linalg.norm((joints_right_pred - joints_right_gt), ord=2, dim=-1)
            joint_right_loss = joint_right_loss.detach().cpu().numpy()
            joints_loss['right'].append(joint_right_loss)
            joints_xyz_list['right'].append(joints_right_pred.detach().cpu().numpy())
            joints_xyz_gt_list['right'].append(joints_right_gt.detach().cpu().numpy())

            vert_left_loss = torch.linalg.norm((verts_left_pred - verts_left_gt), ord=2, dim=-1)
            vert_left_loss = vert_left_loss.detach().cpu().numpy()
            verts_loss['left'].append(vert_left_loss)

            vert_right_loss = torch.linalg.norm((verts_right_pred - verts_right_gt), ord=2, dim=-1)
            vert_right_loss = vert_right_loss.detach().cpu().numpy()
            verts_loss['right'].append(vert_right_loss)

            verts_2d_left_pred = xyz2uvd(verts_left_pred + root_left_gt, cam)
            verts_2d_right_pred = xyz2uvd(verts_right_pred + root_right_gt, cam)
            joints_2d_left_pred = xyz2uvd(joints_left_pred + root_left_gt, cam)
            joints_2d_right_pred = xyz2uvd(joints_right_pred + root_right_gt, cam)

            vert_2d_left_loss = torch.linalg.norm((verts_2d_left_pred - verts_2d_left_gt), ord=2, dim=-1)
            vert_2d_left_loss = vert_2d_left_loss.detach().cpu().numpy()
            verts_2d_loss['left'].append(vert_2d_left_loss)

            vert_2d_right_loss = torch.linalg.norm((verts_2d_right_pred - verts_2d_right_gt), ord=2, dim=-1)
            vert_2d_right_loss = vert_2d_right_loss.detach().cpu().numpy()
            verts_2d_loss['right'].append(vert_2d_right_loss)

            joint_2d_left_loss = torch.linalg.norm((joints_2d_left_pred - joints_2d_left_gt), ord=2, dim=-1)
            joint_2d_left_loss = joint_2d_left_loss.detach().cpu().numpy()
            joints_2d_loss['left'].append(joint_2d_left_loss)

            joint_2d_right_loss = torch.linalg.norm((joints_2d_right_pred - joints_2d_right_gt), ord=2, dim=-1)
            joint_2d_right_loss = joint_2d_right_loss.detach().cpu().numpy()
            joints_2d_loss['right'].append(joint_2d_right_loss)

            if opt.root_joint == 0:
                root_loss = torch.linalg.norm((gt_offset - rel_root_pred), ord=2, dim=-1)
            else:
                joints_right_pred_ori = joints_right_pred_ori + rel_root_pred
                rel_root_pred = joints_right_pred_ori[:, opt.root_joint:opt.root_joint + 1] - joints_left_pred_ori[:,opt.root_joint:opt.root_joint + 1]
                root_loss = torch.linalg.norm((gt_offset - rel_root_pred), ord=2, dim=-1)

            root_loss = root_loss.detach().cpu().numpy()
            root_loss_list.append(root_loss)

            idx += 1
            # if idx > 20:
            #     break

    joints_loss['left'] = np.concatenate(joints_loss['left'], axis=0)
    joints_loss['right'] = np.concatenate(joints_loss['right'], axis=0)
    verts_loss['left'] = np.concatenate(verts_loss['left'], axis=0)
    verts_loss['right'] = np.concatenate(verts_loss['right'], axis=0)

    joints_2d_loss['left'] = np.concatenate(joints_2d_loss['left'], axis=0)
    joints_2d_loss['right'] = np.concatenate(joints_2d_loss['right'], axis=0)
    verts_2d_loss['left'] = np.concatenate(verts_2d_loss['left'], axis=0)
    verts_2d_loss['right'] = np.concatenate(verts_2d_loss['right'], axis=0)

    joints_xyz_left_pd = np.concatenate(joints_xyz_list['left'], axis=0).reshape([-1, 21 * 3])
    joints_xyz_right_pd = np.concatenate(joints_xyz_list['right'], axis=0).reshape([-1, 21 * 3])
    joints_xyz_left_gt = np.concatenate(joints_xyz_gt_list['left'], axis=0).reshape([-1, 21 * 3])
    joints_xyz_right_gt = np.concatenate(joints_xyz_gt_list['right'], axis=0).reshape([-1, 21 * 3])
    joint_error_left = np.concatenate(joints_loss['left'], axis=0).reshape([-1, 21])
    joint_error_right = np.concatenate(joints_loss['right'], axis=0).reshape([-1, 21])
    verts_error_left = np.concatenate(verts_loss['left'], axis=0).reshape([-1, 778])
    verts_error_right = np.concatenate(verts_loss['right'], axis=0).reshape([-1, 778])
    joint_2d_error_left = np.concatenate(joints_2d_loss['left'], axis=0).reshape([-1, 21])
    joint_2d_error_right = np.concatenate(joints_2d_loss['right'], axis=0).reshape([-1, 21])
    verts_2d_error_left = np.concatenate(verts_2d_loss['left'], axis=0).reshape([-1, 778])
    verts_2d_error_right = np.concatenate(verts_2d_loss['right'], axis=0).reshape([-1, 778])
    root_loss = np.concatenate(root_loss_list, axis=0).reshape([-1])

    np.savetxt(file_folder + '/left_joint.txt', joints_xyz_left_pd * 1000, fmt='%.3f')
    np.savetxt(file_folder + '/right_joint.txt', joints_xyz_right_pd * 1000, fmt='%.3f')
    np.savetxt(file_folder + '/joint_left_error.txt', joint_error_left * 1000, fmt='%.3f')
    np.savetxt(file_folder + '/joint_right_error.txt', joint_error_right * 1000, fmt='%.3f')
    np.savetxt(file_folder + '/mesh_left_error.txt', verts_error_left.mean(-1) * 1000, fmt='%.3f')
    np.savetxt(file_folder + '/mesh_right_error.txt', verts_error_right.mean(-1) * 1000, fmt='%.3f')
    np.savetxt(file_folder + '/joint_2d_left_error.txt', joint_2d_error_left, fmt='%.3f')
    np.savetxt(file_folder + '/joint_2d_right_error.txt', joint_2d_error_right, fmt='%.3f')
    np.savetxt(file_folder + '/mesh_2d_left_error.txt', verts_2d_error_left.mean(-1), fmt='%.3f')
    np.savetxt(file_folder + '/mesh_2d_right_error.txt', verts_2d_error_right.mean(-1), fmt='%.3f')
    np.savetxt(file_folder + '/root_loss.txt', root_loss * 1000, fmt='%.3f')
    np.savetxt(file_folder + '/volume.txt', joint_error_right * 1000, fmt='%.3f')

    joints_mean_loss_left = joints_loss['left'].mean() * 1000
    joints_mean_loss_right = joints_loss['right'].mean() * 1000
    verts_mean_loss_left = verts_loss['left'].mean() * 1000
    verts_mean_loss_right = verts_loss['right'].mean() * 1000
    joints_2d_mean_loss_left = joints_2d_loss['left'].mean()
    joints_2d_mean_loss_right = joints_2d_loss['right'].mean()
    verts_2d_mean_loss_left = verts_2d_loss['left'].mean()
    verts_2d_mean_loss_right = verts_2d_loss['right'].mean()
    root_mean_loss = root_loss.mean() * 1000

    print('joint mean error:')
    print('    left: {} mm, right: {} mm'.format(joints_mean_loss_left, joints_mean_loss_right))
    print('    all: {} mm'.format((joints_mean_loss_left + joints_mean_loss_right) / 2))
    print('vert mean error:')
    print('    left: {} mm, right: {} mm'.format(verts_mean_loss_left, verts_mean_loss_right))
    print('    all: {} mm'.format((verts_mean_loss_left + verts_mean_loss_right) / 2))
    print('pixel joint mean error:')
    print('    left: {} mm, right: {} mm'.format(joints_2d_mean_loss_left, joints_2d_mean_loss_right))
    print('    all: {} mm'.format((joints_2d_mean_loss_left + joints_2d_mean_loss_right) / 2))
    print('pixel vert mean error:')
    print('    left: {} mm, right: {} mm'.format(verts_2d_mean_loss_left, verts_2d_mean_loss_right))
    print('    all: {} mm'.format((verts_2d_mean_loss_left + verts_2d_mean_loss_right) / 2))
    print('root error: {} mm'.format(root_mean_loss))
