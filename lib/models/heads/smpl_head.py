import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from einops import rearrange


from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ..components.pose_transformer import TransformerDecoder
from .position_encoding import build_position_encoding


def build_smpl_head(cfg):
    smpl_head_type = cfg.MODEL.SMPL_HEAD.get('TYPE', 'hmr')
    if smpl_head_type == 'transformer_decoder':
        return SMPLTransformerDecoderHead(cfg)
    if smpl_head_type == 'fcn':
        return SMPLFCNHead(cfg)
    if smpl_head_type == 'transformer_decoder_token':
        return SMPLTransformerDecoderTokenHead(cfg)
    if smpl_head_type == 'fcn_fusion':
        return SMPLFCNFusionHead(cfg)
    else:
        raise ValueError('Unknown SMPL head type: {}'.format(smpl_head_type))


class SMPLTransformerDecoderHead(nn.Module):
    """ Cross-attention based SMPL Transformer decoder
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.n_views = cfg.DATASET.N_VIEWS
        self.joint_rep_type = cfg.MODEL.SMPL_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.SMPL.NUM_BODY_JOINTS)
        self.npose = npose
        self.input_is_mean_shape = cfg.MODEL.SMPL_HEAD.get(
            'TRANSFORMER_INPUT', 'zero') == 'mean_shape'
        if cfg.MODEL.BACKBONE.TYPE == 'resnet':
            transformer_args = dict(
                num_tokens=1,
                token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
                dim=1024,
            )
        else:
            transformer_args = dict(
                num_tokens=1,
                token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
                dim=1024,
            )
        transformer_args = {**transformer_args, **
                            dict(cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER)}
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        if cfg.MODEL.SMPL_HEAD.POSITIONAL_ENCODING == 'SinePositionalEncoding3D':
            # TODO find a better way of exposing other arguments
            self.position_embedding = build_position_encoding(cfg)
        dim = transformer_args['dim']
        context_dim = transformer_args['context_dim']
        # self.decpose = nn.Linear(dim, npose)
        # self.decglobalorientation = nn.Linear(context_dim, 6)
        # self.decshape = nn.Linear(dim, 10)
        # self.deccam = nn.Linear(context_dim, 3)

        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)

        self.fc1 = nn.Linear(context_dim, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decglobalorientation = nn.Linear(1024, 6)
        self.deccam = nn.Linear(1024, 3)



        # self.avgpool = nn.AvgPool2d(8, stride=1)
        if cfg.MODEL.BACKBONE.TYPE == 'resnet' or cfg.MODEL.BACKBONE.TYPE == 'swin_v2':
            self.avgpool = nn.AvgPool2d(8, stride=1)
        elif cfg.MODEL.BACKBONE.TYPE == 'vit':
            self.avgpool = nn.AvgPool2d((16, 12), stride=1)

        if cfg.MODEL.SMPL_HEAD.get('INIT_DECODER_XAVIER', False):
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decglobalorientation.weight, gain=0.01)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_pose = torch.from_numpy(
            mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(
            mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(
            mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_global_orientation', init_pose[:, :6])
        self.register_buffer('init_body_pose', init_pose[:, 6:])
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, features, n_views, **kwargs):
        features_pooled = self.avgpool(features)
        features_pooled = features_pooled.view(features_pooled.size(0), -1) # (8, 2048)
        # features = features.view(features.size(0), -1, features.size(1)) # (8, 64, 2048)
        # features = rearrange(features, "(n b) c h w-> b n c h w", n=self.n_views) # (2, 256, 2048)
        features = rearrange(features, "(n b) c h w-> b n c h w", n=n_views) # (2, 256, 2048)

        if self.cfg.MODEL.SMPL_HEAD.POSITIONAL_ENCODING == 'SinePositionalEncoding3D':
            pos_embed = self.position_embedding(features)
            features = features + pos_embed
        features = rearrange(features, "b n c h w-> b (n h w) c") # (512, 2048, 8, 8)
        
        
        # batch_size = xf_b.shape[0]
        n_sample = features_pooled.shape[0] # 8 = 4 * 2
        # batch_size = n_sample // self.n_views # 2
        batch_size = n_sample // n_views # 2
        init_body_pose = self.init_body_pose.expand(batch_size, -1) # (2, 138)
        init_betas = self.init_betas.expand(batch_size, -1) # (2, 10)
        init_cam = self.init_cam.expand(n_sample, -1) # (8, 3)
        init_global_orientation = self.init_global_orientation.expand(
            n_sample, -1) # (8, 6)

        # TODO: Convert init_body_pose to aa rep if needed
        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_global_orientation = init_global_orientation
  
        for i in range(self.cfg.MODEL.SMPL_HEAD.get('IEF_ITERS', 1)):
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_body_pose, pred_betas, pred_cam], dim=1)[
                    :, None, :]
            else:
                token = torch.zeros(batch_size, 1, 1).to(features.device)

            # Pass through transformer
            token_out = self.transformer(token, context=features)
            token_out = token_out.squeeze(1)  # (B, C)

            # Readout from token_out
            pred_body_pose = self.decpose(token_out) + pred_body_pose
            pred_betas = self.decshape(token_out) + pred_betas

            xf = self.fc1(features_pooled)
            xf = self.drop1(xf)
            xf = self.fc2(xf)
            xf = self.drop2(xf)
            pred_global_orientation = self.decglobalorientation(xf) + pred_global_orientation
            pred_cam = self.deccam(xf) + pred_cam

            # pred_global_orientation = self.decglobalorientation(
            #     features_pooled) + pred_global_orientation
            # pred_cam = self.deccam(features_pooled) + pred_cam
            

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(
            batch_size, self.cfg.SMPL.NUM_BODY_JOINTS, 3, 3)
        pred_global_orientation = joint_conversion_fn(pred_global_orientation).view(
            n_sample, 1, 3, 3)

        return pred_body_pose, pred_betas, pred_global_orientation, pred_cam

class SMPLFCNHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_views = cfg.DATASET.N_VIEWS
        # self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.joint_rep_type = cfg.MODEL.SMPL_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.SMPL.NUM_BODY_JOINTS)
        self.npose = npose

        self.fc1 = nn.Linear(self.cfg.MODEL.SMPL_HEAD.IN_CHANNELS * self.n_views + npose + 10 + 6 * self.n_views + 3 * self.n_views, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3 * self.n_views)
        self.decglobalorientation = nn.Linear(1024, 6 * self.n_views)
        if cfg.MODEL.BACKBONE.TYPE == 'resnet' or cfg.MODEL.BACKBONE.TYPE == 'swin_v2':
            self.avgpool = nn.AvgPool2d(8, stride=1)
        elif cfg.MODEL.BACKBONE.TYPE == 'vit':
            self.avgpool = nn.AvgPool2d((16, 12), stride=1)
        if cfg.MODEL.SMPL_HEAD.get('INIT_DECODER_XAVIER', False):
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decglobalorientation.weight, gain=0.01)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_pose = torch.from_numpy(
            mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(
            mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(
            mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_global_orientation', init_pose[:, :6])
        self.register_buffer('init_body_pose', init_pose[:, 6:])
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)
    def forward(self, features, n_iter=3):
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        features = rearrange(features, "(n b) c -> b (n c)", n=self.n_views)
        # n_sample = features.shape[0] # 8 = 4 * 2
        batch_size = features.size(0) # 2
        init_body_pose = self.init_body_pose.expand(batch_size, -1) # (2, 138)
        init_betas = self.init_betas.expand(batch_size, -1) # (2, 10)
        init_cam = self.init_cam.expand(batch_size * self.n_views, -1).contiguous(
    ).view(batch_size, -1) # (8, 3)
        init_global_orientation = self.init_global_orientation.expand(
            batch_size * self.n_views, -1).contiguous(
    ).view(batch_size, -1) # (8, 6)
        # TODO: Convert init_body_pose to aa rep if needed
        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_global_orientation = init_global_orientation
        for i in range(n_iter):
            xc = torch.cat([features, pred_body_pose, pred_betas, pred_global_orientation, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_body_pose = self.decpose(xc) + pred_body_pose
            pred_betas = self.decshape(xc) + pred_betas
            pred_global_orientation = self.decglobalorientation(xc) + pred_global_orientation
            pred_cam = self.deccam(xc) + pred_cam
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(
            batch_size, self.cfg.SMPL.NUM_BODY_JOINTS, 3, 3)
        
        pred_global_orientation = rearrange(pred_global_orientation, "b (n c) -> (n b) c", n=self.n_views)
        pred_global_orientation = joint_conversion_fn(pred_global_orientation).view(
            batch_size * self.n_views, 1, 3, 3)
        pred_cam = rearrange(pred_cam, "b (n c) -> (n b) c", n=self.n_views)
        return pred_body_pose, pred_betas, pred_global_orientation, pred_cam

class SMPLTransformerDecoderTokenHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_views = cfg.DATASET.N_VIEWS
        self.joint_rep_type = cfg.MODEL.SMPL_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.SMPL.NUM_BODY_JOINTS)
        self.npose = npose
        self.input_is_mean_shape = cfg.MODEL.SMPL_HEAD.get(
            'TRANSFORMER_INPUT', 'zero') == 'mean_shape'
        if cfg.MODEL.BACKBONE.TYPE == 'resnet':
            transformer_args = dict(
                num_tokens=1 + self.n_views,
                token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
                dim=1024,
            )
        else:
            transformer_args = dict(
                num_tokens=1 + self.n_views,
                token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
                dim=1024,
            )
        transformer_args = {**transformer_args, **
                            dict(cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER)}
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        if cfg.MODEL.SMPL_HEAD.POSITIONAL_ENCODING == 'SinePositionalEncoding3D':
            # TODO find a better way of exposing other arguments
            self.position_embedding = build_position_encoding(cfg)
        dim = transformer_args['dim']
        context_dim = transformer_args['context_dim']
        self.decpose = nn.Linear(dim, npose)
        self.decglobalorientation = nn.Linear(dim, 6)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)
        # self.avgpool = nn.AvgPool2d(8, stride=1)
        if cfg.MODEL.BACKBONE.TYPE == 'resnet' or cfg.MODEL.BACKBONE.TYPE == 'swin_v2':
            self.avgpool = nn.AvgPool2d(8, stride=1)
        elif cfg.MODEL.BACKBONE.TYPE == 'vit':
            self.avgpool = nn.AvgPool2d((16, 12), stride=1)

        if cfg.MODEL.SMPL_HEAD.get('INIT_DECODER_XAVIER', False):
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decglobalorientation.weight, gain=0.01)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_pose = torch.from_numpy(
            mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(
            mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(
            mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_global_orientation', init_pose[:, :6])
        self.register_buffer('init_body_pose', init_pose[:, 6:])
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, features, **kwargs):
        # features_pooled = self.avgpool(features)
        # features_pooled = features_pooled.view(features_pooled.size(0), -1) # (8, 2048)
        # features = features.view(features.size(0), -1, features.size(1)) # (8, 64, 2048)
        n_sample = features.shape[0]
        features = rearrange(features, "(n b) c h w-> b n c h w", n=self.n_views) # (2, 256, 2048)

        if self.cfg.MODEL.SMPL_HEAD.POSITIONAL_ENCODING == 'SinePositionalEncoding3D':
            pos_embed = self.position_embedding(features)
            features = features + pos_embed
        features = rearrange(features, "b n c h w-> b (n h w) c") # (512, 2048, 8, 8)
        
        
        # batch_size = xf_b.shape[0]
         # 8 = 4 * 2
        batch_size = n_sample // self.n_views # 2
        init_body_pose = self.init_body_pose.expand(batch_size, -1) # (2, 138)
        init_betas = self.init_betas.expand(batch_size, -1) # (2, 10)
        init_cam = self.init_cam.expand(n_sample, -1) # (8, 3)
        init_global_orientation = self.init_global_orientation.expand(
            n_sample, -1) # (8, 6)

        # TODO: Convert init_body_pose to aa rep if needed
        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_global_orientation = init_global_orientation
  
        for i in range(self.cfg.MODEL.SMPL_HEAD.get('IEF_ITERS', 1)):
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_body_pose, pred_betas, pred_cam], dim=1)[
                    :, None, :]
            else:
                token = torch.zeros(batch_size, 1 + self.n_views, 1).to(features.device)

            # Pass through transformer
            token_out = self.transformer(token, context=features)
            # token_out = token_out.squeeze(1)  # (B, C)

            # Readout from token_out
            pred_body_pose = self.decpose(token_out[:, 1]) + pred_body_pose
            pred_betas = self.decshape(token_out[:, 1]) + pred_betas
            token_cam = rearrange(token_out[:, 1:], "b n c -> (n b) c")
            pred_global_orientation = self.decglobalorientation(
                token_cam) + pred_global_orientation
            pred_cam = self.deccam(token_cam) + pred_cam
            

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(
            batch_size, self.cfg.SMPL.NUM_BODY_JOINTS, 3, 3)
        pred_global_orientation = joint_conversion_fn(pred_global_orientation).view(
            n_sample, 1, 3, 3)

        return pred_body_pose, pred_betas, pred_global_orientation, pred_cam
    
class SMPLFCNFusionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_views = cfg.DATASET.N_VIEWS
        self.joint_rep_type = cfg.MODEL.SMPL_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.SMPL.NUM_BODY_JOINTS)
        self.npose = npose
        self.in_channels = cfg.MODEL.SMPL_HEAD.IN_CHANNELS
        self.fc1 = nn.Linear(self.cfg.MODEL.SMPL_HEAD.IN_CHANNELS + npose + 10, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)

        
        self.fc3 = nn.Linear(self.in_channels, 1024)
        self.drop3 = nn.Dropout()
        self.fc4 = nn.Linear(1024, 1024)
        self.drop4 = nn.Dropout()

        self.decglobalorientation = nn.Linear(1024, 6)
        self.deccam = nn.Linear(1024, 3)

        self.attn = nn.Linear(self.in_channels, self.in_channels)
        # self.drop3 = nn.Dropout()
        # self.avgpool = nn.AvgPool2d(8, stride=1)
        if cfg.MODEL.BACKBONE.TYPE == 'resnet' or cfg.MODEL.BACKBONE.TYPE == 'swin_v2':
            self.avgpool = nn.AvgPool2d(8, stride=1)
        elif cfg.MODEL.BACKBONE.TYPE == 'vit':
            self.avgpool = nn.AvgPool2d((16, 12), stride=1)

        if cfg.MODEL.SMPL_HEAD.get('INIT_DECODER_XAVIER', False):
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decglobalorientation.weight, gain=0.01)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_pose = torch.from_numpy(
            mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(
            mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(
            mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_global_orientation', init_pose[:, :6])
        self.register_buffer('init_body_pose', init_pose[:, 6:])
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)

    def forward(self, features, n_iter=3):
        n_sample = features.shape[0] # 8 = 4 * 2
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        features_attn = self.attn(features)
        features_views = rearrange(features, "(n b) c -> b n c", n=self.n_views)
        features_attn = rearrange(features_attn, "(n b) c -> b n c", n=self.n_views)
        features_attn = F.softmax(features_attn, dim=1)
        features_fuse = torch.sum(features_views * features_attn, dim=1)
        # n_sample = features.shape[0] # 8 = 4 * 2
        batch_size = features_fuse.size(0) # 2
        init_body_pose = self.init_body_pose.expand(batch_size, -1) # (2, 138)
        init_betas = self.init_betas.expand(batch_size, -1) # (2, 10)
        init_cam = self.init_cam.expand(n_sample, -1) # (8, 3)
        init_global_orientation = self.init_global_orientation.expand(
            n_sample, -1) # (8, 6)

        # TODO: Convert init_body_pose to aa rep if needed
        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_global_orientation = init_global_orientation
        for i in range(n_iter):
            xc = torch.cat([features_fuse, pred_body_pose, pred_betas], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_body_pose = self.decpose(xc) + pred_body_pose
            pred_betas = self.decshape(xc) + pred_betas

            xf = self.fc3(features)
            xf = self.drop3(xf)
            xf = self.fc4(xf)
            xf = self.drop4(xf)
            pred_global_orientation = self.decglobalorientation(xf) + pred_global_orientation
            pred_cam = self.deccam(xf) + pred_cam

            # pred_global_orientation = self.decglobalorientation(features) + pred_global_orientation
            # pred_cam = self.deccam(features) + pred_cam
  
        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        
        pred_body_pose = joint_conversion_fn(pred_body_pose).view(
            batch_size, self.cfg.SMPL.NUM_BODY_JOINTS, 3, 3)
        pred_global_orientation = joint_conversion_fn(pred_global_orientation).view(
            n_sample, 1, 3, 3)

        return pred_body_pose, pred_betas, pred_global_orientation, pred_cam