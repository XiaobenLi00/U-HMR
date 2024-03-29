# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pickle
import collections
import torchvision.transforms as transforms

from .joints_dataset import JointsDataset


class MPII3D(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        }

        if cfg.DATASET.CROP:
            anno_file = osp.join(self.root, 'mpi_inf_3dhp', 'annot',
                                 'mpi_inf_3dhp_{}_new.pkl'.format(image_set))
        else:
            anno_file = osp.join(self.root, 'mpi_inf_3dhp', 'annot',
                                 'mpi_inf_3dhp_{}_uncrop.pkl'.format(image_set))

        self.db = self.load_db(anno_file)

        # self.u2a_mapping = super().get_mapping()
        # super().do_mapping()

        # self.grouping = self.get_group(self.db)
        # self.group_size = len(self.grouping)
        self.db_size = len(self.db)

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = db[i]['camera_id']
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1, -1]
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        if self.is_train:
            filtered_grouping = filtered_grouping[::5]
            # filtered_grouping = filtered_grouping[::500]
        else:
            filtered_grouping = filtered_grouping[::5]
            # filtered_grouping = filtered_grouping[::640]
            # filtered_grouping = filtered_grouping[::64]

        # if self.is_train:
        #     filtered_grouping = filtered_grouping[:1]
        # else:
        #     # pass
        #     filtered_grouping = filtered_grouping[46635:46636]
        #     # filtered_grouping = filtered_grouping[46636:46637]
        #     # filtered_grouping = filtered_grouping[:1]



        return filtered_grouping

    def __getitem__(self, idx):
        input, meta = [], []
        i, m = super().__getitem__(idx)
        input.append(i)
        meta.append(m)
        return input, meta
        # input, target, weight, meta = [], [], [], []
        # input, meta = [], []
        # items = self.grouping[idx]
        # for item in items:
        #     # i, t, w, m = super().__getitem__(item)
        #     i, m = super().__getitem__(item)
        #     input.append(i)
        #     # target.append(t)
        #     # weight.append(w)
        #     meta.append(m)
        # # return input, target, weight, meta
        # return input, meta

    def __len__(self):
        # return self.group_size
        return self.db_size

    def get_key_str(self, datum):
        return 's_{:02}_seq_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['sequence'],
            datum['image_id'])

    def evaluate(self, pred, *args, **kwargs):
        pass