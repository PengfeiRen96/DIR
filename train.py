import os
import shutil
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from config import cfg
from utils.logger import setup_logger
from dataset.interhand import InterHandDataset
from models.dir import DIR
from utils.visualize import draw_2d_skeleton
import cv2


def vis(inputs, targets, outs, id, all_sample=False, stage_num=0):
    img = inputs['img_rgb'].numpy()
    joints_left_gt = targets['joint_2d_left'].numpy() * 128 + 128
    joints_right_gt = targets['joint_2d_right'].numpy() * 128 + 128

    if len(joints_left_gt.shape) == 4:
        img = img.reshape([-1, 256, 256, 3])
        joints_left_gt = joints_left_gt.reshape([-1, 21, 3])
        joints_right_gt = joints_right_gt.reshape([-1, 21, 3])

        sample_list = [0, 1, 2, 3]
        for index in sample_list:
            if 'pd_joint_uv_left' in outs:
                img_left = draw_2d_skeleton(img[index], joints_left_gt[index])
                cv2.imwrite(cfg.output_root + '/vis/%d_left_gt.png' % (id * cfg.batch_size + index), img_left)

                img_right = draw_2d_skeleton(img[index], joints_right_gt[index])
                cv2.imwrite(cfg.output_root + '/vis/%d_right_gt.png' % (id * cfg.batch_size + index), img_right)
                joints_left_pd = outs['pd_joint_uv_left'].detach().cpu().numpy() * 128 + 128
                joints_right_pd = outs['pd_joint_uv_right'].detach().cpu().numpy() * 128 + 128
                img_left = draw_2d_skeleton(img[index], joints_left_pd[index])
                cv2.imwrite(cfg.output_root + '/vis/%d_left_pd_%d.png' % (id * cfg.batch_size + index, stage_num), img_left)
                img_right = draw_2d_skeleton(img[index], joints_right_pd[index])
                cv2.imwrite(cfg.output_root + '/vis/%d_right_pd_%d.png' % (id * cfg.batch_size + index, stage_num), img_right)
    else:
        sample_list = [0]
        for index in sample_list:
            if 'pd_joint_uv_left' in outs:
                img_left = draw_2d_skeleton(img[index], joints_left_gt[index])
                cv2.imwrite(cfg.output_root + '/vis/%d_left_gt.png' % (id * cfg.batch_size + index), img_left)

                img_right = draw_2d_skeleton(img[index], joints_right_gt[index])
                cv2.imwrite(cfg.output_root + '/vis/%d_right_gt.png' % (id * cfg.batch_size + index), img_right)
                joints_left_pd = outs['pd_joint_uv_left'].detach().cpu().numpy() * 128 + 128
                joints_right_pd = outs['pd_joint_uv_right'].detach().cpu().numpy() * 128 + 128
                img_left = draw_2d_skeleton(img[index], joints_left_pd[index])
                cv2.imwrite(cfg.output_root + '/vis/%d_left_pd_%d.png' % (id * cfg.batch_size + index, stage_num), img_left)
                img_right = draw_2d_skeleton(img[index], joints_right_pd[index])
                cv2.imwrite(cfg.output_root + '/vis/%d_right_pd_%d.png' % (id * cfg.batch_size + index, stage_num), img_right)


def train():
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer._make_model()
    trainer._make_batch_loader()
    min_error = 100

    for epoch in range(trainer.start_epoch, cfg.total_epoch):
        for iteration, (inputs, targets, meta_infos) in tqdm(enumerate(trainer.trian_loader)):
            trainer.optimizer.zero_grad()
            outs_list, loss = trainer.model(inputs, targets, meta_infos)
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            if iteration % cfg.print_iter == 0:
                screen = ['[Epoch %d/%d]' % (epoch, cfg.total_epoch),
                          '[Batch %d/%d]' % (iteration, len(trainer.trian_loader)),
                          '[lr %f]' % (trainer.get_lr())]
                screen += ['[%s: %.4f]' % ('loss_' + k, v.detach()) for k, v in loss.items()]
                trainer.logger.info(''.join(screen))
            if iteration % cfg.draw_iter == 0:
                if len(outs_list) > 1:
                    for stage_index, outs in enumerate(outs_list):
                        vis(inputs, targets, outs, iteration, stage_num=stage_index)
                else:
                    vis(inputs, targets, outs_list[-1], iteration, stage_num=0)

        trainer.schedule.step()
        trainer.save_model(trainer.model, trainer.optimizer, trainer.schedule, epoch, 'latest')

        if not epoch % cfg.eval_interval:
            error = trainer.test_model()
            if error < min_error:
                trainer.save_model(trainer.model, trainer.optimizer, trainer.schedule, epoch, 'best')
                min_error = error


def test():
    torch.backends.cudnn.benchmark = True
    tester = Tester()
    tester._make_model()
    tester._make_batch_loader()
    tester.model.eval()
    tester.test_model()


class Trainer:
    def __init__(self):
        log_folder = os.path.join(cfg.output_root, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'train_' + cfg.experiment_name + '.log')

        vis_folder = os.path.join(cfg.output_root, 'vis')
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)

        file_folder = os.path.join(cfg.output_root, 'files')
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        shutil.copytree('./SemGCN', file_folder + '/SemGCN/')
        shutil.copytree('./models', file_folder + '/models/')
        shutil.copytree('./dataset', file_folder + '/dataset/')
        shutil.copytree('./utils', file_folder + '/utils/')
        shutil.copy('./train.py', file_folder + '/train.py')
        shutil.copy('./config.py', file_folder + '/config.py')

        self.logger = setup_logger(output=logfile, name="Training")
        self.logger.info('Start training: %s' % ('train_' + cfg.experiment_name))

    def load_model(self, checkpoint_dir, model, optimizer, schedule):
        checkpoint = torch.load(checkpoint_dir)
        self.logger.info("Loading the model of epoch-{} from {}...".format(checkpoint['last_epoch'], checkpoint_dir))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        schedule.load_state_dict(checkpoint['schedule'])
        start_epoch = checkpoint['last_epoch'] + 1
        self.logger.info('The model is loaded successfully.')
        return start_epoch, model

    def save_model(self, model, optimizer, schedule, epoch, name):
        save = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'schedule': schedule.state_dict(),
            'last_epoch': epoch
        }
        path_checkpoint = os.path.join(cfg.output_root, 'checkpoint')
        if not os.path.exists(path_checkpoint):
            os.makedirs(path_checkpoint)
        save_path = os.path.join(path_checkpoint, "%s.pth" % (name))
        torch.save(save, save_path)
        self.logger.info('Save checkpoint to {}'.format(save_path))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    @torch.no_grad()
    def test_model(self):
        self.model.eval()
        iter_num = 0
        joint_error_left_sum, joint_error_right_sum = np.zeros([cfg.stage_num]), np.zeros([cfg.stage_num])
        mesh_error_left_sum, mesh_error_right_sum = np.zeros([cfg.stage_num]), np.zeros([cfg.stage_num])
        for iteration, (inputs, targets, meta_infos) in tqdm(enumerate(self.test_loader)):
            outs_list, loss = self.model(inputs, targets, meta_infos)
            for stage_index in range(cfg.stage_num):
                joint_error_left, joint_error_right, mesh_error_left, mesh_error_right = \
                    self.test_dataset.evaluate(outs_list[stage_index], targets, meta_infos)
                joint_error_left_sum[stage_index] += joint_error_left
                joint_error_right_sum[stage_index] += joint_error_right
                mesh_error_left_sum[stage_index] += mesh_error_left
                mesh_error_right_sum[stage_index] += mesh_error_right
                if iteration % cfg.draw_iter == 0:
                    vis(inputs, targets, outs_list[-1], iteration, False, stage_index)

            iter_num += 1


        for stage_index in range(cfg.stage_num):
            joints_mean_loss_left = joint_error_left_sum[stage_index] / iter_num
            joints_mean_loss_right = joint_error_right_sum[stage_index] / iter_num
            verts_mean_loss_left = mesh_error_left_sum[stage_index] / iter_num
            verts_mean_loss_right = mesh_error_right_sum[stage_index] / iter_num

            print('MPJPE_%d:' % (stage_index))
            print('    left: {} mm, right: {} mm'.format(joints_mean_loss_left, joints_mean_loss_right))
            print('    all: {} mm'.format((joints_mean_loss_left + joints_mean_loss_right) / 2))
            print('MPVPE_%d:' % (stage_index))
            print('    left: {} mm, right: {} mm'.format(verts_mean_loss_left, verts_mean_loss_right))
            print('    all: {} mm'.format((verts_mean_loss_left + verts_mean_loss_right) / 2))

            self.logger.info(
                'MPJPE_{}: left {} mm, right {} mm, AVG {} mm'.format(
                    stage_index,
                    joints_mean_loss_left, joints_mean_loss_right,
                    (joints_mean_loss_left + joints_mean_loss_right) / 2))
            self.logger.info(
                'MPVPE_{}: left {} mm, right {} mm, AVG {} mm'.format(
                    stage_index,
                    verts_mean_loss_left, verts_mean_loss_right,
                    (verts_mean_loss_left + verts_mean_loss_right) / 2))

        self.model.train()
        return (joints_mean_loss_left + joints_mean_loss_right) / 2

    def _make_batch_loader(self):
        self.logger.info("Creating dataset...")

        self.train_dataset = InterHandDataset(cfg.data_dir, 'train', cfg.root_joint)
        self.trian_loader = DataLoader(self.train_dataset,
                                       batch_size=cfg.batch_size,
                                       num_workers=cfg.num_worker,
                                       shuffle=True,
                                       pin_memory=True,
                                       drop_last=True)
        self.test_dataset = InterHandDataset(cfg.data_dir, 'test', cfg.root_joint)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=cfg.batch_size,
                                      num_workers=cfg.num_worker,
                                      shuffle=False,
                                      pin_memory=True,
                                      drop_last=True)
        self.logger.info("The dataset is created successfully.")

    def _make_model(self):
        self.logger.info("Making the model...")
        model = DIR(cfg.joint_num, cfg.mano_path, cfg.root_joint).cuda()

        optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': cfg.lr}], cfg.lr)

        if cfg.lr_scheduler == 'cosine':
            schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.total_epoch, eta_min=0)
        elif cfg.lr_scheduler == 'step':
            schedule = optim.lr_scheduler.MultiStepLR(optimizer, [30], gamma=0.1, last_epoch=-1)

        if cfg.continue_train:
            start_epoch, model = self.load_model(cfg.checkpoint, model, optimizer, schedule)
        else:
            start_epoch = 0
        model.train()
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.logger.info("The model is made successfully.")


class Tester:
    def __init__(self):
        log_folder = os.path.join(cfg.output_root, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'eval_' + cfg.experiment_name + '.log')
        self.logger = setup_logger(output=logfile, name="Evaluation")
        self.logger.info('Start evaluation: %s' % ('eval_' + cfg.experiment_name))

        vis_folder = os.path.join(cfg.output_root, 'vis')
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)

    def _make_batch_loader(self):
        self.logger.info("Creating dataset...")
        self.dataset = InterHandDataset(cfg.data_dir, 'test', cfg.root_joint)
        self.loader = DataLoader(self.dataset,
                                 batch_size=cfg.batch_size,
                                 num_workers=cfg.num_worker,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=True)
        self.logger.info("The dataset is created successfully.")


    def load_model(self, model):
        if cfg.checkpoint != '':
            self.logger.info('Loading the model from {}...'.format(cfg.checkpoint))
            checkpoint = torch.load(cfg.checkpoint)
            model.load_state_dict(checkpoint['net'])
            self.logger.info('The model is loaded successfully.')
        elif cfg.output_root != '':
            self.logger.info('Loading the model from {}...'.format(cfg.output_root))
            checkpoint = torch.load(cfg.output_root+'/checkpoint/latest.pth')
            model.load_state_dict(checkpoint['net'])
            self.logger.info('The model is loaded successfully.')
        else:
            self.logger.info('No model is loaded.')
        return model

    def _make_model(self):
        self.logger.info("Making the model...")
        model = DIR(cfg.joint_num, cfg.mano_path, cfg.root_joint).cuda()
        model = self.load_model(model)
        model.eval()
        self.model = model
        self.logger.info("The model is made successfully.")

    @torch.no_grad()
    def test_model(self):
        self.model.eval()
        iter_num = 0
        joint_error_left_sum, joint_error_right_sum = np.zeros([cfg.stage_num]), np.zeros([cfg.stage_num])
        mesh_error_left_sum, mesh_error_right_sum = np.zeros([cfg.stage_num]), np.zeros([cfg.stage_num])
        for iteration, (inputs, targets, meta_infos) in tqdm(enumerate(self.loader)):
            outs_list, loss = self.model(inputs, targets, meta_infos)
            for stage_index in range(cfg.stage_num):
                joint_error_left, joint_error_right, mesh_error_left, mesh_error_right = \
                    self.dataset.evaluate(outs_list[stage_index], targets, meta_infos)
                joint_error_left_sum[stage_index] += joint_error_left
                joint_error_right_sum[stage_index] += joint_error_right
                mesh_error_left_sum[stage_index] += mesh_error_left
                mesh_error_right_sum[stage_index] += mesh_error_right
                if iteration % cfg.draw_iter == 0:
                    vis(inputs, targets, outs_list[-2], iteration, False, stage_index)

            iter_num += 1

        for stage_index in range(cfg.stage_num):
            joints_mean_loss_left = joint_error_left_sum[stage_index] / iter_num
            joints_mean_loss_right = joint_error_right_sum[stage_index] / iter_num
            verts_mean_loss_left = mesh_error_left_sum[stage_index] / iter_num
            verts_mean_loss_right = mesh_error_right_sum[stage_index] / iter_num

            print('MPJPE_%d:' % (stage_index))
            print('    left: {} mm, right: {} mm'.format(joints_mean_loss_left, joints_mean_loss_right))
            print('    all: {} mm'.format((joints_mean_loss_left + joints_mean_loss_right) / 2))
            print('MPVPE_%d:' % (stage_index))
            print('    left: {} mm, right: {} mm'.format(verts_mean_loss_left, verts_mean_loss_right))
            print('    all: {} mm'.format((verts_mean_loss_left + verts_mean_loss_right) / 2))

            self.logger.info(
                'MPJPE_{}: left {} mm, right {} mm, AVG {} mm'.format(
                    stage_index,
                    joints_mean_loss_left, joints_mean_loss_right,
                    (joints_mean_loss_left + joints_mean_loss_right) / 2))
            self.logger.info(
                'MPVPE_{}: left {} mm, right {} mm, AVG {} mm'.format(
                    stage_index,
                    verts_mean_loss_left, verts_mean_loss_right,
                    (verts_mean_loss_left + verts_mean_loss_right) / 2))

if __name__ == '__main__':
    if cfg.phase == 'train':
        train()
    else:
        test()
