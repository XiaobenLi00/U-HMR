import torch
import torch.nn as nn
import numpy as np 
from .smpl_wrapper import SMPL
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from typing import Any, Dict, Mapping, Tuple
from ..utils.geometry import perspective_projection, rot6d_to_rotmat, aa_to_rotmat
from ..utils import vis
from ..utils.pose_utils import reconstruction_error
from einops import rearrange
from .discriminator import Discriminator
import torch.nn.functional as F
import math
from tensorboardX import SummaryWriter
import logging
from .backbones import create_backbone
from .heads import build_smpl_head
import trimesh
logger = logging.getLogger(__name__)

class Mv_Fusion(nn.Module):
    def __init__(self, cfg, tensorboard_log_dir) -> None:
        super(Mv_Fusion, self).__init__()
        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            logger.info(f'=> Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            # self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'])
            if cfg.MODEL.BACKBONE.TYPE == 'resnet':
                self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu'), strict=False)
            elif cfg.MODEL.BACKBONE.TYPE == 'vit':
                self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'])
                # self.backbone.eval()
                # for param in self.backbone.parameters():
                    # param.requires_grad = False
            elif cfg.MODEL.BACKBONE.TYPE == 'swin_v2':
                self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['model'])

        # Create SMPL head
        self.smpl_head = build_smpl_head(cfg)

        # Create discriminator
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smpl_parameter_loss = ParameterLoss()
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.optimizer, self.optimizer_disc = self.get_optimizer()
        else:
            self.optimizer = self.get_optimizer()


        smpl_cfg = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)
        joints_list_17 = [14, 2, 1, 0, 3, 4, 5, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]
        self.joints_list = [i + 25 for i in joints_list_17]
        # self.n_views = cfg.DATASET.N_VIEWS

        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        if cfg.TRAIN.RENDER_MESH:
            
            from ..utils.renderer import Renderer
            self.mesh_renderer = Renderer(focal_length=cfg.EXTRA.FOCAL_LENGTH,
                        img_res=256, faces=self.smpl.faces)
        self.step_count = {'train': 0,
                        'val': 0,
                        'train_vis': 0,
                        'val_vis': 0}
        self.log_dict_mpii3d = {'mpjpe': [],
                                'pa-mpjpe': []}
        # self.epoch = 0
        # self.len_data = 0
        # self.count_dict = {'epoch': 0, 'len_data': 0}

    def forward_step(self, x, n_views)  -> Dict:
        x = torch.cat(x, dim = 0)
        if self.cfg.MODEL.BACKBONE.TYPE == 'vit':
            features = self.backbone(x[:,:,:,32:-32])
        else:
            features = self.backbone(x)
        pred_body_pose, pred_betas, pred_global_orientation, pred_cam = self.smpl_head(features, n_views)
        n_sample = x.shape[0]

        # pred_body_pose = pred_body_pose.repeat(self.n_views, 1, 1, 1)
        # pred_betas = pred_betas.repeat(self.n_views, 1)
        pred_body_pose = pred_body_pose.repeat(n_views, 1, 1, 1)
        pred_betas = pred_betas.repeat(n_views, 1)
        pred_pose = torch.cat([pred_global_orientation, pred_body_pose], dim = 1)
        
        output = self.forward_smpl(pred_pose, pred_betas, pred_cam, n_sample)

        return output
    
    def compute_loss(self, output, meta, dataset = None):
        
        pred_smpl_params = output['pred_smpl_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']

        n_sample = pred_smpl_params['body_pose'].shape[0]

        # Get annotations
        # gt_keypoints_2d = batch['keypoints_2d']
        # gt_keypoints_3d = batch['keypoints_3d']
        gt_2d_list = []
        gt_3d_list = []
        vis_list = []
        for m in meta:
            gt_2d_list.append(m['joints_2d_transformed'])
            gt_3d_list.append(m['joints_3d_camera'])
            vis_list.append(m['joints_vis'])
        vis_joints = torch.cat(vis_list, dim=0)
        gt_keypoints_2d = torch.cat(gt_2d_list, dim=0)
        gt_keypoints_3d = torch.cat(gt_3d_list, dim=0) / 1000
        gt_keypoints_2d = gt_keypoints_2d / (self.cfg.MODEL.IMAGE_SIZE[0] / 1.) - 0.5
        gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, [0], :]
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, [0], :]
        # print(pred_keypoints_3d.shape)
        # print(gt_keypoints_3d.shape)
        # print(vis_joints.shape)
        # print(pred_keypoints_2d.shape) 
        # print(gt_keypoints_2d.shape)
        # Compute 2D and 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(vis_joints[:,:,[0]]*pred_keypoints_2d, vis_joints[:,:,[0]]*gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(vis_joints[:,:,[0]]*pred_keypoints_3d, vis_joints[:,:,[0]]*gt_keypoints_3d)
        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d
        # if has_smpl:
        # gt_smpl_params = batch['smpl_params']
        gt_global_orient_list = []

        gt_body_pose_list = []
        gt_shape_list = []
        has_global_orient_list = []

        has_body_pose_list = []
        has_shape_list = []
        
        for m in meta:
            # gt_global_orient_list.append(aa_to_rotmat(m['smpl_params']['global_orient']))
            # gt_body_pose_list.append(aa_to_rotmat(m['smpl_params']['body_pose']))
            gt_global_orient_list.append(m['smpl_params']['global_orient'])
            gt_body_pose_list.append(m['smpl_params']['body_pose'])
            gt_shape_list.append(m['smpl_params']['betas'])

            has_global_orient_list.append(m['has_smpl_params']['global_orient'])
            has_body_pose_list.append(m['has_smpl_params']['body_pose'])
            has_shape_list.append(m['has_smpl_params']['betas'])
        gt_global_orient = torch.cat(gt_global_orient_list, dim=0)
        gt_body_pose = torch.cat(gt_body_pose_list, dim=0)
        gt_shape = torch.cat(gt_shape_list, dim=0)
        has_global_orient = torch.cat(has_global_orient_list, dim=0)
        has_body_pose = torch.cat(has_body_pose_list, dim=0)
        has_shape = torch.cat(has_shape_list, dim=0)

        gt_smpl_params = {'global_orient': gt_global_orient,
                        'body_pose': gt_body_pose,
                        'betas': gt_shape}
        has_smpl_params = {'global_orient':has_global_orient,
                            'body_pose': has_body_pose,
                        'betas': has_shape}
        # Compute loss on SMPL parameters
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            gt = gt_smpl_params[k].view(n_sample, -1)
            has_gt = has_smpl_params[k]
            loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(n_sample, -1), gt.reshape(n_sample, -1), has_gt)

        loss += sum([loss_smpl_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smpl_params])

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach())
        # if has_smpl:
        for k, v in loss_smpl_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses

        # mpjpe = np.mean(np.sqrt(np.sum((pred_keypoints_3d.detach().cpu().numpy() - gt_keypoints_3d.detach().cpu().numpy()) ** 2, axis=-1))) * 1000
        # rec_error = reconstruction_error(pred_keypoints_3d.detach().cpu().numpy(), gt_keypoints_3d.detach().cpu().numpy(), reduction='mean') * 1000

        mpjpe =((np.sqrt(np.sum((pred_keypoints_3d.detach().cpu().numpy() - gt_keypoints_3d.detach().cpu().numpy()) ** 2, axis=-1)) * vis_joints[:,:,0].detach().cpu().numpy()).sum() / (vis_joints[:,:,0]+1e-9).detach().cpu().numpy().sum() ) * 1000
        
        if dataset == 'totalcapture':
            idx = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16]
            rec_error = reconstruction_error(pred_keypoints_3d[:, idx].detach().cpu().numpy(), gt_keypoints_3d[:, idx].detach().cpu().numpy(), reduction='mean') * 1000
        else:
            rec_error = reconstruction_error(pred_keypoints_3d.detach().cpu().numpy(), gt_keypoints_3d.detach().cpu().numpy(), reduction='mean') * 1000

        self.log_dict_mpii3d['mpjpe'].append(mpjpe)
        self.log_dict_mpii3d['pa-mpjpe'].append(rec_error)

        metrics = dict(mpjpe=mpjpe, rec_error=rec_error)
        output['metrics'] = metrics
        return loss
    
    def training_step_discriminator(self, batch: Dict,
                                    body_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            body_pose (torch.Tensor): Regressed body pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = body_pose.shape[0]
        gt_body_pose = batch['body_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss_disc.detach()
    
    def forward(self, x, meta, batch_idx, mocap, meters, len_data, n_views, epoch = 0, train: bool = True, dataset = None):
        output = self.forward_step(x, n_views)
        pred_smpl_params = output['pred_smpl_params']
        loss = self.compute_loss(output, meta, dataset)
        n_samples = mocap['body_pose'].shape[0]
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(n_samples, -1), pred_smpl_params['betas'].reshape(n_samples, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / n_samples
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
                loss_disc = self.training_step_discriminator(mocap, pred_smpl_params['body_pose'].reshape(n_samples, -1), pred_smpl_params['betas'].reshape(n_samples, -1), self.optimizer_disc)
                output['losses']['loss_gen'] = loss_adv
                output['losses']['loss_disc'] = loss_disc
        self.tensorboard_logging(x, output, self.step_count, batch_idx, meters, len_data, epoch, train)
        if not train:
            return output['metrics']
        return None


    def get_optimizer(self)  -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:

        all_params = list(self.backbone.parameters()) + list(self.smpl_head.parameters())
        param_groups = [{'params': filter(lambda p: p.requires_grad, all_params), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        # lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                                lr=self.cfg.TRAIN.LR,
                                                                                                # lr=self.cfg.TRAIN.LR * 10,
                                                weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
            return optimizer, optimizer_disc
        return optimizer
    
    def tensorboard_logging(self, input, output, step_count, batch_idx, meters, len_data, epoch, train: bool = True):

        mode = 'train' if train else 'val'
        losses = output['losses']
        # n_samples = output['pred_keypoints_2d'].shape[0]
        for loss_name, val in losses.items():
            self.writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count['{}'.format(mode)])
        meters['{}_loss'.format(mode)].update(output['losses']['loss'])
        meters['{}_mpjpe'.format(mode)].update(output['metrics']['mpjpe'])
        meters['{}_rec_error'.format(mode)].update(output['metrics']['rec_error'])
        step_count['{}'.format(mode)] += 1
        if batch_idx % self.cfg.TRAIN.LOG_INTERVAL == 0:
            # images = torch.cat(input, dim=0)
            # images = rearrange(images, "(n b) c d e -> (b n) c d e", n=4)
            # pred_keypoints_2d = rearrange(output['pred_keypoints_2d'], "(n b) c d -> (b n) c d", n=self.n_views)
            # keypoints_2d_vis = vis.visualize_2d_pose(images, pred_keypoints_2d)
            # self.writer.add_image(mode + '/pred_2d', keypoints_2d_vis, step_count['{}_vis'.format(mode)])
            if self.cfg.TRAIN.RENDER_MESH:
                images = images.detach() * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
                images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
                pred_vertice = rearrange(output['pred_vertices'], "(n b) c d -> (b n) c d", n=4)
                pred_cam_t = rearrange(output['pred_cam_t'], "(n b) c -> (b n) c", n=4)
                images_pred = self.mesh_renderer.visualize_tb(pred_vertice.detach(), pred_cam_t.detach(), images)
                self.writer.add_image(mode + '/pred_shape', images_pred, step_count['{}_vis'.format(mode)])
                # mesh_vertices = rearrange(pred_vertice, "(b n) c d -> b n c d", n=4)
                # for b in range(mesh_vertices.shape[0]):
                #     for n in range(mesh_vertices.shape[1]):
                #         mesh_vertice = mesh_vertices.clone().detach().cpu().numpy()[b, n]
                #         vertex_colors = np.ones([mesh_vertice.shape[0], 4]) * [0.82, 0.9, 0.98, 1.0]
                #         face_colors = np.ones([self.smpl.faces.shape[0], 4]) * [0.82, 0.9, 0.98, 1.0]
                #         mesh = trimesh.Trimesh(mesh_vertice, self.smpl.faces, face_colors=face_colors, vertex_colors=vertex_colors, process=False)
                #         mesh.export('/home/benlee/projects/HPE/multi_view_trans/output/h36m' + '/meshes/' + 'mesh_{}_{}_{}.obj'.format(step_count['{}_vis'.format(mode)] ,b, n))
            step_count['{}_vis'.format(mode)] += 1
            if train:
                msg = (f'Epoch: [{epoch}][{batch_idx}/{len_data}]\t'
                f'Loss: {meters["{}_loss".format(mode)].val:.5f} ({meters["{}_loss".format(mode)].avg:.5f})\t'
                f'MPJPE: {meters["{}_mpjpe".format(mode)].val:.3f} ({meters["{}_mpjpe".format(mode)].avg:.3f})\t'
                f'REC_ERROR: {meters["{}_rec_error".format(mode)].val:.3f} ({meters["{}_rec_error".format(mode)].avg:.3f})')
            else:
                msg = (f'Test: [{batch_idx}/{len_data}]\t'
                f'Loss: {meters["{}_loss".format(mode)].val:.5f} ({meters["{}_loss".format(mode)].avg:.5f})\t'
                f'MPJPE: {meters["{}_mpjpe".format(mode)].val:.3f} ({meters["{}_mpjpe".format(mode)].avg:.3f})\t'
                f'REC_ERROR: {meters["{}_rec_error".format(mode)].val:.3f} ({meters["{}_rec_error".format(mode)].avg:.3f})')
            logger.info(msg)
    
    def forward_smpl(self, pred_rotmat, pred_betas, pred_cam, n_sample):

        pred_smpl_params = {'global_orient': pred_rotmat[:, [0]],
                            'body_pose': pred_rotmat[:, 1:],
                            'betas': pred_betas}
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = pred_smpl_params

        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(n_sample, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE[0] * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(n_sample, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(n_sample, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(n_sample, -1)
        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints[:, self.joints_list, :] # (8, 44, 3) -> (8, 17, 3)
        pred_vertices = smpl_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(n_sample, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(n_sample, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE[0]) # range [-0.5, 0.5]
        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(n_sample, -1, 2)
        return output

if __name__ == "__main__":
    from ..utils.config import get_config
    from ..utils.log_utils import create_logger
    cfg_name = 'swin.yaml'
    cfg = get_config('../../experiments/h36m/{}'.format(cfg_name), merge= False)
    if cfg.IS_TRAIN:
        phase = 'train'
    else:
        phase = 'test'
    logger, final_output_dir, tensorboard_log_dir = create_logger(cfg, cfg_name, phase)
    model = Mv_Fusion(cfg, tensorboard_log_dir)
    x = [torch.randn(2, 3, 256, 256) for i in range(4)]