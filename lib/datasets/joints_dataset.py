# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import random
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset
import sys

from ..utils import zipreader
from ..utils.transforms import get_affine_transform
from ..utils.transforms import affine_transform
from ..utils.triangulate import triangulate_poses, camera_to_world_frame


class JointsDataset(Dataset):

    def __init__(self, cfg, subset, is_train, transform=None):
        self.is_train = is_train
        self.subset = subset

        self.root = cfg.DATASET.ROOT
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.transform = transform
        self.db = []
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.num_joints = 17
    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self, ):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_dir = 'images.zip@' if self.data_format == 'zip' else ''
        image_file = osp.join(self.root, db_rec['source'], image_dir, 'images',
                              db_rec['image'])
        if self.data_format == 'zip':

            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        joints = db_rec['joints_2d'].copy()
        joints_vis = db_rec['joints_vis'].copy()

        center = np.array(db_rec['center']).copy()
        scale = np.array(db_rec['scale']).copy()
        rotation = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0
        # sf = 0.2
        # scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        # center_shift = np.array([(random.random()-0.5)*40 , (random.random()-0.5)*40])
        # center += center_shift
        trans = get_affine_transform(center, scale, rotation, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                if (np.min(joints[i, :2]) < 0 or
                        joints[i, 0] >= self.image_size[0] or
                        joints[i, 1] >= self.image_size[1]):
                    joints_vis[i, :] = 0

        meta = {
            'scale': scale,
            'center': center,
            'rotation': rotation,
            'joints_2d': db_rec['joints_2d'].astype(np.float32),
            'joints_2d_transformed': joints.astype(np.float32),
            # 'joints_3d_world': camera_to_world_frame(db_rec['joints_3d_camera'], db_rec['camera']['R'], db_rec['camera']['T']).astype(np.float32),
            'joints_3d_camera': db_rec['joints_3d_camera'].astype(np.float32),
            # 'camera_params': db_rec['camera'],
            'joints_vis': joints_vis,
            'source': db_rec['source'],
            'index': idx
        }
        # if self.is_train and db_rec['source'] == 'h36m':
        if self.is_train:
            meta['smpl_params'] = {'global_orient': db_rec['global_orient'].astype(np.float32),
                            'body_pose': db_rec['body_pose'].astype(np.float32),
                            'betas': db_rec['betas'].astype(np.float32)}
            meta['has_smpl_params'] = {'global_orient': np.ones(1,dtype=np.float32)[0],
                            'body_pose': np.ones(1,dtype=np.float32)[0],
                            'betas': np.ones(1,dtype=np.float32)[0]}
        else:
            # meta['has_smpl'] = False
            meta['smpl_params'] = {'global_orient': np.zeros((1,3,3),dtype=np.float32),
                            'body_pose': np.zeros((23,3,3),dtype=np.float32),
                            'betas': np.zeros(10,dtype=np.float32)}
            meta['has_smpl_params'] = {'global_orient': np.zeros(1,dtype=np.float32)[0],
                            'body_pose': np.zeros(1,dtype=np.float32)[0],
                            'betas': np.zeros(1,dtype=np.float32)[0]}
        return input, meta