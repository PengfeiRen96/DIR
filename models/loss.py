import torch
import torch.nn as nn
from torch.nn import functional as F


class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = torch.LongTensor(face).cuda()

    def forward(self, coord_out, coord_gt, mask=None):
        v1_out = coord_out[:, self.face[:, 1], :] - coord_out[:, self.face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, self.face[:, 2], :] - coord_out[:, self.face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, self.face[:, 2], :] - coord_out[:, self.face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, self.face[:, 1], :] - coord_gt[:, self.face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, self.face[:, 2], :] - coord_gt[:, self.face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        if mask is not None:
            return (loss*mask).sum() / (mask.sum()+1e-8)
        else:
            return loss.mean()


class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = torch.LongTensor(face).cuda()

    def forward(self, coord_out, coord_gt, mask=None):
        d1_out = torch.sqrt(
            torch.sum((coord_out[:, self.face[:, 0], :] - coord_out[:, self.face[:, 1], :]) ** 2, 2, keepdim=True) + 1e-12)
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, self.face[:, 0], :] - coord_out[:, self.face[:, 2], :]) ** 2, 2, keepdim=True) + 1e-12)
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, self.face[:, 1], :] - coord_out[:, self.face[:, 2], :]) ** 2, 2, keepdim=True) + 1e-12)

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, self.face[:, 0], :] - coord_gt[:, self.face[:, 1], :]) ** 2, 2, keepdim=True) + 1e-12)
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, self.face[:, 0], :] - coord_gt[:, self.face[:, 2], :]) ** 2, 2, keepdim=True) + 1e-12)
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, self.face[:, 1], :] - coord_gt[:, self.face[:, 2], :]) ** 2, 2, keepdim=True) + 1e-12)

        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        loss = torch.cat((diff1, diff2, diff3), 1)
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            return loss.mean()


class SmoothL1Loss(torch.nn.Module):
    def __init__(self, size_average=False):
        super(SmoothL1Loss, self).__init__()
        self.size_average = size_average

    def forward(self, x, y, mask=None):
        batch_size = x.size(0)
        total_loss = 0
        assert(x.shape == y.shape)
        x = x.reshape(batch_size, -1)
        y = y.reshape(batch_size, -1)
        z = (x - y).float()
        mse_mask = (torch.abs(z) < 0.01).float()
        l1_mask = (torch.abs(z) >= 0.01).float()
        mse = mse_mask * z
        l1 = l1_mask * z
        total_loss += torch.mean(self._calculate_MSE(mse)*mse_mask, dim=-1)
        total_loss += torch.mean(self._calculate_L1(l1)*l1_mask, dim=-1)

        if mask is not None:
            return (total_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            return total_loss.mean()

    def _calculate_MSE(self, z):
        return 0.5 * (torch.pow(z, 2))

    def _calculate_L1(self, z):
        return 0.01 * (torch.abs(z) - 0.005)
