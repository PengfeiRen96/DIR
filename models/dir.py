import os
import math
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
from models.backbone.hourglass import Residual
from models.loss import NormalVectorLoss, EdgeLengthLoss, SmoothL1Loss
from models.backbone.resnet import resnet50 as ResNet50
from utils.utils import projection_batch_xy
from models.lovasz_loss import lovasz_softmax
from SemGCN.utils import adj_mx_from_edges, get_sketch_setting
from SemGCN.p_gcn import ResSimplePGCN
from transformer.mixSTE import STE
from manopth.manolayer import ManoLayer as ObmanManoLayer


# Joint Space Interaction + Project Joint Feature to Image Space
class Joint2BoneFeature(nn.Module):
    def __init__(self, img_feat_dim, emd_dim, joint_dim, joint_num, feature_size, mano_pth, root_joint, distance=1):
        super(Joint2BoneFeature, self).__init__()
        edge = get_sketch_setting()
        adj = adj_mx_from_edges(joint_num, edge, sparse=False, eye=False)
        self.bone_num = 20
        self.parent = torch.Tensor([0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]).long()
        self.child = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]).long()
        self.gcn_left = ResSimplePGCN(adj, emd_dim, num_layers=4)
        self.gcn_right = ResSimplePGCN(adj, emd_dim, num_layers=4)
        self.img2joint_left = ImgFeature2JointFeature(img_feat_dim, emd_dim)
        self.img2joint_right = ImgFeature2JointFeature(img_feat_dim, emd_dim)
        self.pos_emb_left = nn.Sequential(
            nn.Conv1d(3, emd_dim, 1),
            nn.BatchNorm1d(emd_dim),
            nn.ReLU(),
            nn.Conv1d(emd_dim, emd_dim, 1)
        )
        self.pos_emb_right = nn.Sequential(
            nn.Conv1d(3, emd_dim, 1),
            nn.BatchNorm1d(emd_dim),
            nn.ReLU(),
            nn.Conv1d(emd_dim, emd_dim, 1)
        )
        self.global_pos_emb = nn.Sequential(
            nn.Conv1d(3, emd_dim, 1),
            nn.BatchNorm1d(emd_dim),
            nn.ReLU(),
            nn.Conv1d(emd_dim, emd_dim, 1)
        )

        self.interaction = STE(num_joints=joint_num * 2, in_chans=emd_dim, out_dim=joint_dim, depth=4)
        self.proj_feat_emb = nn.Sequential(
            nn.Conv1d(joint_dim, joint_dim, 1),
            nn.BatchNorm1d(joint_dim),
            nn.ReLU(),
            nn.Conv1d(joint_dim, joint_dim, 1)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(joint_dim * self.bone_num * 2, img_feat_dim, 3, 1, 1),
            nn.BatchNorm2d(img_feat_dim),
            nn.ReLU(),
            nn.Conv2d(img_feat_dim, img_feat_dim, 1)
        )

        self.regressor = RegressorOffset(joint_num * joint_dim, mano_pth, root_joint)

        x = (torch.arange(feature_size) + 0.5)
        y = (torch.arange(feature_size) + 0.5)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        img_gird = torch.stack((grid_y, grid_x), dim=-1).reshape([feature_size ** 2, 2]).contiguous()
        self.register_buffer('img_gird', img_gird)
        self.joint_dim = joint_dim
        self.joint_num = joint_num
        self.feature_size = feature_size
        self.distance = distance
        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img_feat,
                joint_xyz_left, joint_xyz_right,
                joint_uv_left, joint_uv_right,
                pre_mano_para_left, pre_mano_para_right,
                offset):

        B = joint_xyz_left.size(0)

        joint_img_feat_left = self.img2joint_left(img_feat, joint_uv_left).reshape(B, -1, self.joint_num).permute(0, 2, 1).contiguous()
        joint_img_feat_right = self.img2joint_right(img_feat, joint_uv_right).reshape(B, -1,self.joint_num).permute(0, 2, 1).contiguous()

        joint_pos_feat_left = self.pos_emb_left(joint_xyz_left.permute(0, 2, 1).contiguous() / 0.15).permute(0, 2, 1).contiguous()
        joint_pos_feat_right = self.pos_emb_right(joint_xyz_right.permute(0, 2, 1).contiguous() / 0.15).permute(0, 2, 1).contiguous()

        joint_feat_left = joint_pos_feat_left + joint_img_feat_left
        joint_feat_right = joint_pos_feat_right + joint_img_feat_right

        joint_feat_left = self.gcn_left(joint_feat_left)
        joint_feat_right = self.gcn_right(joint_feat_right)

        joint_global_pos_feat_left = self.global_pos_emb((joint_xyz_left / 0.15 - offset / 2).permute(0, 2, 1)).permute(0, 2, 1)
        joint_global_pos_feat_right = self.global_pos_emb((joint_xyz_right / 0.15 + offset / 2).permute(0, 2, 1)).permute(0, 2, 1)

        joint_feat_left = joint_feat_left + joint_global_pos_feat_left
        joint_feat_right = joint_feat_right + joint_global_pos_feat_right

        joint_feat = torch.cat((joint_feat_left, joint_feat_right), dim=1)
        joint_feat = self.interaction(joint_feat)
        joint_feat_left, joint_feat_right = torch.split(joint_feat, [self.joint_num, self.joint_num], dim=1)

        result = self.regressor(joint_feat_left, joint_feat_right, pre_mano_para_left, pre_mano_para_right, offset)

        joint_feat_left = self.proj_feat_emb(joint_feat_left.permute(0,2,1)).permute(0,2,1)
        joint_feat_right = self.proj_feat_emb(joint_feat_right.permute(0,2,1)).permute(0,2,1)
        img_feat_left = self.bone_proj(result["pd_joint_uv_left"], joint_feat_left)
        img_feat_right = self.bone_proj(result["pd_joint_uv_right"], joint_feat_right)
        img_feat = self.fusion(torch.cat((img_feat_left, img_feat_right), dim=1))

        feat_list = {
            'img_feat': img_feat,
            'joint_feat_left': joint_feat_left,
            'joint_feat_right': joint_feat_right,
            'vis_img_feat': img_feat_left+img_feat_right
        }
        return result, feat_list

    def lineseg_dists(self, p, a, b):
        device = p.device
        d_ba = b - a
        d = torch.divide(d_ba, (torch.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

        s = torch.multiply(a - p, d).sum(dim=1)
        t = torch.multiply(p - b, d).sum(dim=1)

        h = torch.maximum(torch.maximum(s, t), torch.zeros(s.size(0)).to(device))
        d_pa = p - a
        c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

        return torch.hypot(h, c)

    def bone_proj(self, joint_uv, joint_feat):
        device = joint_uv.device
        B, J, C = joint_feat.size()
        S = self.feature_size
        joint_uv = (joint_uv+1)/2*S

        bone_a = torch.index_select(joint_uv, dim=1, index=self.parent.to(device)).reshape([B, 1, self.bone_num, 2]).repeat(1, S ** 2, 1, 1)
        bone_b = torch.index_select(joint_uv, dim=1, index=self.child.to(device)).reshape([B, 1, self.bone_num, 2]).repeat(1, S ** 2, 1, 1)
        bone_a, bone_b = bone_a.reshape([-1, 2]), bone_b.reshape([-1, 2])

        feat_a = torch.index_select(joint_feat, dim=1, index=self.parent.to(device))
        feat_b = torch.index_select(joint_feat, dim=1, index=self.child.to(device))
        feat_a, feat_b = feat_a.reshape([B, 1, self.bone_num, -1]), feat_b.reshape([B, 1, self.bone_num, -1])

        img_gird = self.img_gird.reshape([1, S ** 2, 1, 2]).repeat(B, 1, self.bone_num, 1).reshape(-1, 2)
        distance = self.lineseg_dists(img_gird, bone_a, bone_b)
        distance = distance.reshape([B, S ** 2, self.bone_num])
        mask = distance.lt(self.distance)
        dist_a = F.pairwise_distance(img_gird, bone_a, p=2)
        dist_b = F.pairwise_distance(img_gird, bone_b, p=2)
        weight_a = 1 - torch.divide(dist_a, dist_a + dist_b)
        weight_b = 1 - torch.divide(dist_b, dist_a + dist_b)
        weight_a, weight_b = weight_a.reshape([B, -1, self.bone_num, 1]), weight_b.reshape([B, -1, self.bone_num, 1])

        img_feat = feat_a * weight_a + feat_b * weight_b
        mask = mask.reshape([B, -1, self.bone_num, 1])
        img_feat = torch.where(mask, img_feat, torch.zeros_like(img_feat))
        img_feat = img_feat.reshape([B, S, S, self.bone_num*C]).permute(0, 3, 1, 2)
        return img_feat


class ImgFeature2JointFeature(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ImgFeature2JointFeature, self).__init__()
        self.filters = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, 1),
        )
        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img_feature, joint_uv):
        sampled_mesh_feat = F.grid_sample(img_feature, joint_uv.unsqueeze(1).detach()).squeeze(-2)
        sampled_mesh_feat = self.filters(sampled_mesh_feat)
        return sampled_mesh_feat.reshape(img_feature.size(0), -1).contiguous()


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class InitRegressor(nn.Module):
    def __init__(self, feat_dim, mano_path, root_joint):
        super(InitRegressor, self).__init__()
        self.mano_layer_right = ObmanManoLayer(root_rot_mode='6D', joint_rot_mode='axisang', use_pca=True, mano_root=mano_path,
                                               side='right', ncomps=45, center_idx=root_joint, flat_hand_mean=False, robust_rot=True)
        self.mano_layer_left = ObmanManoLayer(root_rot_mode='6D', joint_rot_mode='axisang', use_pca=True, mano_root=mano_path,
                                              side='left', ncomps=45, center_idx=root_joint, flat_hand_mean=False, robust_rot=True)
        self.fix_shape(self.mano_layer_left, self.mano_layer_right)

        self.attention_left = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim//2, 3, 1, 1),
            nn.BatchNorm2d(feat_dim//2),
            nn.ReLU(),
            nn.Conv2d(feat_dim//2, 1, 1, 1),
            nn.Sigmoid()
        )

        self.attention_right = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim//2, 3, 1, 1),
            nn.BatchNorm2d(feat_dim//2),
            nn.ReLU(),
            nn.Conv2d(feat_dim//2, 1, 1, 1),
            nn.Sigmoid()
        )

        self.offset = nn.Linear(feat_dim, 3)
        self.mano_left = nn.Linear(feat_dim, 3 * 2 + 15 * 3 + 10 + 3)
        self.mano_right = nn.Linear(feat_dim, 3 * 2 + 15 * 3 + 10 + 3)
        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)


    def forward(self, feat):
        batch_size = feat.size(0)

        attn_left = self.attention_left(feat)
        feat_left = (feat * attn_left).sum(-1).sum(-1) / (attn_left.sum(-1).sum(-1)+1e-8)
        attn_right = self.attention_right(feat)
        feat_right = (feat * attn_right).sum(-1).sum(-1) / (attn_right.sum(-1).sum(-1)+1e-8)

        pd_offset = self.offset(feat.mean(-1).mean(-1))
        mano_para_left = self.mano_left(feat_left)
        mano_para_right = self.mano_right(feat_right)

        pd_mano_pose_left, pd_mano_beta_left, para_left = torch.split(mano_para_left, [6 + 15 * 3, 10, 3], dim=-1)
        pd_mano_pose_right, pd_mano_beta_right, para_right = torch.split(mano_para_right, [6 + 15 * 3, 10, 3], dim=-1)
        pd_mesh_xyz_left, pd_joint_xyz_left = self.mano_layer_left(pd_mano_pose_left, pd_mano_beta_left)
        pd_mesh_xyz_right, pd_joint_xyz_right = self.mano_layer_right(pd_mano_pose_right, pd_mano_beta_right)

        pd_joint_uv_left = projection_batch_xy(para_left[:, 0], para_left[:, 1:], pd_joint_xyz_left)
        pd_mesh_uv_left = projection_batch_xy(para_left[:, 0], para_left[:, 1:], pd_mesh_xyz_left)
        pd_joint_uv_right = projection_batch_xy(para_right[:, 0], para_right[:, 1:], pd_joint_xyz_right)
        pd_mesh_uv_right = projection_batch_xy(para_right[:, 0], para_right[:, 1:], pd_mesh_xyz_right)

        output = {
            'pd_rel_joint': None,
            'pd_offset': pd_offset,
            'pd_mano_pose_left': pd_mano_pose_left,
            'pd_mano_pose_right': pd_mano_pose_right,
            'pd_mano_beta_left': pd_mano_beta_left,
            'pd_mano_beta_right': pd_mano_beta_right,
            'pd_proj_left': para_left,
            'pd_proj_right': para_right,
            'pd_mano_para_left': mano_para_left,
            'pd_mano_para_right': mano_para_right,
            'pd_joint_uv_left': pd_joint_uv_left,
            'pd_joint_uv_right': pd_joint_uv_right,
            'pd_mesh_uv_left': pd_mesh_uv_left,
            'pd_mesh_uv_right': pd_mesh_uv_right,
            'pd_joint_xyz_left': pd_joint_xyz_left,
            'pd_joint_xyz_right': pd_joint_xyz_right,
            'pd_mesh_xyz_left': pd_mesh_xyz_left,
            'pd_mesh_xyz_right': pd_mesh_xyz_right,
            'pd_var_left': None,
            'pd_var_right': None,
            'rel_joint_right': None
        }
        return output
    def fix_shape(self, mano_layer_left, mano_layer_right):
        if torch.sum(torch.abs(mano_layer_left.th_shapedirs[:, 0, :] - mano_layer_right.th_shapedirs[:, 0, :])) < 1:
            print('Fix shapedirs bug of MANO')
            mano_layer_left.th_shapedirs[:, 0, :] *= -1


class RegressorOffset(nn.Module):
    def __init__(self, feat_dim, mano_path, root_joint):
        super(RegressorOffset, self).__init__()
        self.mano_layer_right = ObmanManoLayer(root_rot_mode='6D', joint_rot_mode='axisang', use_pca=True, mano_root=mano_path,
                                               side='right', ncomps=45, center_idx=root_joint, flat_hand_mean=False, robust_rot=True)
        self.mano_layer_left = ObmanManoLayer(root_rot_mode='6D', joint_rot_mode='axisang', use_pca=True, mano_root=mano_path,
                                              side='left', ncomps=45, center_idx=root_joint, flat_hand_mean=False, robust_rot=True)
        self.fix_shape(self.mano_layer_left, self.mano_layer_right)

        mano_para_dim = 3 * 2 + 15 * 3 + 10 + 3

        self.mano_left = nn.Linear(feat_dim + mano_para_dim, mano_para_dim)
        self.mano_right = nn.Linear(feat_dim + mano_para_dim, mano_para_dim)
        self.offset = nn.Linear(feat_dim * 2 + 3, 3)
        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)

    def forward(self, sampled_feat_l, sampled_feat_r, mano_para_l_init, mano_para_r_init, offset_init):
        B, J, C = sampled_feat_l.size()

        sampled_feat_l = sampled_feat_l.reshape([B, J * C]).contiguous()
        sampled_feat_r = sampled_feat_r.reshape([B, J * C]).contiguous()
        global_feat_l = torch.cat((sampled_feat_l, mano_para_l_init.detach()), dim=-1)
        global_feat_r = torch.cat((sampled_feat_r, mano_para_r_init.detach()), dim=-1)

        global_feat = torch.cat((sampled_feat_l, sampled_feat_r, offset_init.squeeze(1)), dim=-1)
        pd_offset = self.offset(global_feat)

        pd_mano_para_left = self.mano_left(global_feat_l)
        pd_mano_para_right = self.mano_right(global_feat_r)

        pd_mano_pose_left, pd_mano_beta_left, pd_para_left = torch.split(pd_mano_para_left, [2 * 3 + 15 * 3, 10, 3], dim=-1)
        pd_mano_pose_right, pd_mano_beta_right, pd_para_right = torch.split(pd_mano_para_right, [2 * 3 + 15 * 3, 10, 3], dim=-1)
        pd_mesh_xyz_left, pd_joint_xyz_left = self.mano_layer_left(pd_mano_pose_left, pd_mano_beta_left)
        pd_mesh_xyz_right, pd_joint_xyz_right = self.mano_layer_right(pd_mano_pose_right, pd_mano_beta_right)

        pd_joint_uv_left = projection_batch_xy(pd_para_left[:, 0], pd_para_left[:, 1:], pd_joint_xyz_left)
        pd_mesh_uv_left = projection_batch_xy(pd_para_left[:, 0], pd_para_left[:, 1:], pd_mesh_xyz_left)
        pd_joint_uv_right = projection_batch_xy(pd_para_right[:, 0], pd_para_right[:, 1:], pd_joint_xyz_right)
        pd_mesh_uv_right = projection_batch_xy(pd_para_right[:, 0], pd_para_right[:, 1:], pd_mesh_xyz_right)

        output = {
            'pd_offset': pd_offset,
            'pd_rel_joint': None,
            'pd_proj_left': pd_para_left,
            'pd_proj_right': pd_para_right,
            'pd_mano_pose_left': pd_mano_pose_left,
            'pd_mano_pose_right': pd_mano_pose_right,
            'pd_mano_para_left': pd_mano_para_left,
            'pd_mano_para_right': pd_mano_para_right,
            'pd_joint_uv_left': pd_joint_uv_left,
            'pd_joint_uv_right': pd_joint_uv_right,
            'pd_mesh_uv_left': pd_mesh_uv_left,
            'pd_mesh_uv_right': pd_mesh_uv_right,
            'pd_joint_xyz_left': pd_joint_xyz_left,
            'pd_joint_xyz_right': pd_joint_xyz_right,
            'pd_mesh_xyz_left': pd_mesh_xyz_left,
            'pd_mesh_xyz_right': pd_mesh_xyz_right,
        }
        return output

    def fix_shape(self, mano_layer_left, mano_layer_right):
        if torch.sum(torch.abs(mano_layer_left.th_shapedirs[:, 0, :] - mano_layer_right.th_shapedirs[:, 0, :])) < 1:
            print('Fix shapedirs bug of MANO')
            mano_layer_left.th_shapedirs[:, 0, :] *= -1


class FusionJointInterIterDecoder(nn.Module):
    def __init__(self, joint_num, mano_pth, root_joint, inDim=[2048, 1024, 512, 256], fDim=[256, 256, 256, 256]):
        super(FusionJointInterIterDecoder, self).__init__()
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.skip_layer4 = Residual(inDim[1], fDim[0])
        self.fusion_layer4 = Residual(inDim[0] + fDim[0], fDim[1])
        self.projecter_4 = Joint2BoneFeature(fDim[1], 128, 64, joint_num, 16, mano_pth, root_joint, distance=1)
        self.enhance_layer4 = Residual(fDim[1] * 2, fDim[1])

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.skip_layer3 = Residual(inDim[2], fDim[1])
        self.fusion_layer3 = Residual(fDim[1] * 2, fDim[2])
        self.projecter_3 = Joint2BoneFeature(fDim[2], 128, 64, joint_num, 32, mano_pth, root_joint, distance=2)
        self.enhance_layer3 = Residual(fDim[2] * 2, fDim[2])

        self.conv_final = nn.Sequential(
            nn.Conv2d(fDim[3], fDim[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(fDim[3]),
            nn.ReLU(True),
            nn.Conv2d(fDim[3], fDim[3], 1, 1)
        )

        self.seg = nn.Sequential(
            nn.Conv2d(in_channels=fDim[3], out_channels=fDim[3] // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(fDim[3] // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=fDim[3] // 2, out_channels=3, kernel_size=1, stride=1))
        self.dense = nn.Sequential(
            nn.Conv2d(in_channels=fDim[3], out_channels=fDim[3] // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(fDim[3] // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=fDim[3] // 2, out_channels=3, kernel_size=1, stride=1))

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x, result_dict):
        c1, c2, c3, c4 = x
        outputs = []

        ########################### Stage 1 ################################
        c4_up = self.up4(c4)
        c3_skip = self.skip_layer4(c3)
        fusion_feat = self.fusion_layer4(torch.cat((c4_up, c3_skip), dim=1))

        refine_result_dict, out_feat = self.projecter_4(fusion_feat,
                                                        result_dict['pd_joint_xyz_left'].detach(),
                                                        result_dict['pd_joint_xyz_right'].detach(),
                                                        result_dict['pd_joint_uv_left'].detach(),
                                                        result_dict['pd_joint_uv_right'].detach(),
                                                        result_dict['pd_mano_para_left'].detach(),
                                                        result_dict['pd_mano_para_right'].detach(),
                                                        result_dict['pd_offset'].detach().unsqueeze(1)
                                                        )
        enhance_feat = self.enhance_layer4(torch.cat((fusion_feat, out_feat['img_feat']), dim=1))
        outputs.append(dict(refine_result_dict, **out_feat))

        ########################### Stage 2 ################################
        c3_up = self.up3(enhance_feat)
        c2_skip = self.skip_layer3(c2)
        fusion_feat = self.fusion_layer3(torch.cat((c3_up, c2_skip), dim=1))
        refine_result_dict, out_feat = self.projecter_3(fusion_feat,
                                                        refine_result_dict['pd_joint_xyz_left'].detach(),
                                                        refine_result_dict['pd_joint_xyz_right'].detach(),
                                                        refine_result_dict['pd_joint_uv_left'].detach(),
                                                        refine_result_dict['pd_joint_uv_right'].detach(),
                                                        refine_result_dict['pd_mano_para_left'].detach(),
                                                        refine_result_dict['pd_mano_para_right'].detach(),
                                                        refine_result_dict['pd_offset'].detach().unsqueeze(1))
        enhance_feat = self.enhance_layer3(torch.cat((fusion_feat, out_feat['img_feat']), dim=1))
        outputs.append(dict(refine_result_dict, **out_feat))


        feat = self.conv_final(enhance_feat)
        seg = self.seg(feat)
        dense = self.dense(feat)
        out_list = {
            'result_list': outputs,
            'seg': seg,
            'dense': dense,
            'proj_feat': out_feat['vis_img_feat'],
        }
        return out_list


class DIR(nn.Module):
    def __init__(self, joint_num, mano_path, root_joint=0):
        super(DIR, self).__init__()
        self.joint_num = joint_num
        weights = ResNet50_Weights.IMAGENET1K_V2
        pretrained_backbone = resnet50(weights=weights)
        self.backbone = ResNet50()

        pretrained_dict = pretrained_backbone.state_dict()
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)

        self.mesh_sample_num = joint_num
        self.init_regressor = InitRegressor(self.backbone.inplanes, mano_path, root_joint)
        self.decoder = FusionJointInterIterDecoder(self.joint_num, mano_path, root_joint)

        self.coord_weight = 10
        self.dense_weight = 1
        self.l1_loss = SmoothL1Loss()
        self.normal_loss_right = NormalVectorLoss(self.init_regressor.mano_layer_right.th_faces)
        self.edge_loss_right = EdgeLengthLoss(self.init_regressor.mano_layer_right.th_faces)
        self.normal_loss_left = NormalVectorLoss(self.init_regressor.mano_layer_left.th_faces)
        self.edge_loss_left = EdgeLengthLoss(self.init_regressor.mano_layer_left.th_faces)
        self.seg_loss = nn.CrossEntropyLoss(weight=torch.Tensor([.1, 0.45, 0.45]))

    def forward(self, input, target, meta_info):
        x = input['img'].cuda()

        feats = self.backbone(x)
        init_out = self.init_regressor(feats[-1])
        decode_list = self.decoder(feats, init_out)
        iter_outs = [init_out] + decode_list['result_list']

        outs_list = []
        for outs in iter_outs:
            outs_list.append(
                {'pd_joint_uv_left': outs['pd_joint_uv_left'],
                 'pd_joint_uv_right': outs['pd_joint_uv_right'],
                 'pd_mesh_xyz_left': outs['pd_mesh_xyz_left'],
                 'pd_mesh_xyz_right': outs['pd_mesh_xyz_right'],
                 'pd_joint_xyz_left': outs['pd_joint_xyz_left'],
                 'pd_joint_xyz_right': outs['pd_joint_xyz_right'],
                 'pd_proj_left': outs['pd_proj_left'],
                 'pd_proj_right': outs['pd_proj_right'],
                 'pd_offset': outs['pd_offset'],
                 'pd_rel_joint': outs['pd_rel_joint'],
                 }
            )
        outs_list.append({
            'dense': decode_list['dense'],
            'seg': decode_list['seg'],
            'proj_feat': decode_list['proj_feat'],
        })
        loss = {}
        if self.training:
            gt_joint_uvd_left = target['joint_2d_left'].cuda()
            gt_joint_uvd_right = target['joint_2d_right'].cuda()
            gt_mesh_uvd_left = target['mesh_2d_left'].cuda()
            gt_mesh_uvd_right = target['mesh_2d_right'].cuda()

            gt_joint_xyz_left = target['joint_3d_left'].cuda()
            gt_joint_xyz_right = target['joint_3d_right'].cuda()
            gt_mesh_xyz_left = target['mesh_3d_left'].cuda()
            gt_mesh_xyz_right = target['mesh_3d_right'].cuda()

            gt_center_left = meta_info['center_left'].cuda()
            gt_center_right = meta_info['center_right'].cuda()

            gt_joint_normal_xyz_left = (gt_joint_xyz_left - gt_center_left) / 0.15
            gt_mesh_normal_xyz_left = (gt_mesh_xyz_left - gt_center_left) / 0.15
            gt_joint_normal_xyz_right = (gt_joint_xyz_right - gt_center_right) / 0.15
            gt_mesh_normal_xyz_right = (gt_mesh_xyz_right - gt_center_right) / 0.15
            gt_offset = (gt_center_right - gt_center_left) / 0.15

            map_size = decode_list['seg'].size(-1)
            gt_seg_id = target['seg'].cuda()
            gt_dense = target['dense'].cuda()
            gt_seg_id_down = F.interpolate(gt_seg_id, (map_size, map_size), mode='nearest').long().squeeze(1)
            gt_dense_down = F.interpolate(gt_dense, (map_size, map_size), mode='bilinear')
            loss['seg'] = self.seg_loss(decode_list['seg'], gt_seg_id_down) * 0.1 * self.dense_weight
            loss['dense'] = self.l1_loss(decode_list['dense'], gt_dense_down) * self.dense_weight
            loss['lovasz'] = lovasz_softmax(decode_list['seg'], gt_seg_id_down) * 0.1 * self.dense_weight

            for index, out in enumerate(iter_outs):
                loss['joint_left_uv_%d' % (index)] = self.l1_loss(out["pd_joint_uv_left"],gt_joint_uvd_left[:, :, :2]) * self.coord_weight
                loss['joint_right_uv_%d' % (index)] = self.l1_loss(out["pd_joint_uv_right"],gt_joint_uvd_right[:, :, :2]) * self.coord_weight
                loss['mesh_left_uv_%d' % (index)] = self.l1_loss(out["pd_mesh_uv_left"],gt_mesh_uvd_left[:, :, :2]) * self.coord_weight
                loss['mesh_right_uv_%d' % (index)] = self.l1_loss(out["pd_mesh_uv_right"],gt_mesh_uvd_right[:, :, :2]) * self.coord_weight

                joints_left_pred, joints_right_pred = out["pd_joint_xyz_left"] / 0.15, out["pd_joint_xyz_right"] / 0.15
                meshs_left_pred, meshs_right_pred = out["pd_mesh_xyz_left"] / 0.15, out["pd_mesh_xyz_right"] / 0.15

                joints_left_pred, joints_right_pred = joints_left_pred, joints_right_pred
                meshs_left_pred, meshs_right_pred = meshs_left_pred, meshs_right_pred

                loss['joint_left_xyz_%d' % (index)] = self.l1_loss(joints_left_pred, gt_joint_normal_xyz_left) * self.coord_weight
                loss['joint_right_xyz_%d' % (index)] = self.l1_loss(joints_right_pred, gt_joint_normal_xyz_right) * self.coord_weight
                loss['mesh_left_xyz_%d' % (index)] = self.l1_loss(meshs_left_pred, gt_mesh_normal_xyz_left) * self.coord_weight
                loss['mesh_right_xyz_%d' % (index)] = self.l1_loss(meshs_right_pred, gt_mesh_normal_xyz_right) * self.coord_weight

                loss['edge_left_%d' % (index)] = self.edge_loss_left(meshs_left_pred, gt_mesh_normal_xyz_left).mean()
                loss['edge_right_%d' % (index)] = self.edge_loss_right(meshs_right_pred, gt_mesh_normal_xyz_right).mean()
                loss['normal_left_%d' % (index)] = self.normal_loss_left(meshs_left_pred, gt_mesh_normal_xyz_left).mean() * 0.1
                loss['normal_right_%d' % (index)] = self.normal_loss_right(meshs_right_pred, gt_mesh_normal_xyz_right).mean() * 0.1

                if out["pd_offset"] is not None:
                    loss['offset_%d' % (index)] = self.l1_loss(out["pd_offset"], gt_offset.squeeze(1))* self.coord_weight

        return outs_list, loss

