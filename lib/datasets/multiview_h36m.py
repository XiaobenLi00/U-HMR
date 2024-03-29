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


class MultiViewH36M(JointsDataset):

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

        if is_train:
            anno_file = osp.join(self.root, 'h36m', 'annot',
                                 'h36m_{}_with_mosh_sample_5_rot.pkl'.format(image_set))
        else:
            anno_file = osp.join(self.root, 'h36m', 'annot',
                                 'h36m_{}.pkl'.format(image_set))

        self.db = self.load_db(anno_file)

        # print(len(self.db))
        # self.u2a_mapping = super().get_mapping()
        # super().do_mapping()
        if not cfg.DATASET.WITH_DAMAGED:
            self.db = [db_rec for db_rec in self.db if not self.isdamaged(db_rec)]
        # print(len(self.db))
        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)
        # print(self.group_size)


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
            # filtered_grouping = filtered_grouping[::400]
            pass
        else:
            filtered_grouping = filtered_grouping[::64]
            # pass



        return filtered_grouping

    def __getitem__(self, idx):
        # input, target, weight, meta = [], [], [], []
        input, meta = [], []
        items = self.grouping[idx]
        for item in items:
            # i, t, w, m = super().__getitem__(item)
            i, m = super().__getitem__(item)
            input.append(i)
            # target.append(t)
            # weight.append(w)
            meta.append(m)
        # return input, target, weight, meta
        # return input, meta, idx
        return input, meta
    def __len__(self):
        return self.group_size
    
    def isdamaged(self, db_rec):
        # from https://github.com/yihui-he/epipolar-transformers/blob/4da5cbca762aef6a89d37f889789f772b87d2688/data/datasets/joints_dataset.py#L174
        #damaged seq
        #'Greeting-2', 'SittingDown-2', 'Waiting-1'
        if db_rec['subject'] == 9:
            if db_rec['action'] != 5 or db_rec['subaction'] != 2:
                if db_rec['action'] != 10 or db_rec['subaction'] != 2:
                    if db_rec['action'] != 13 or db_rec['subaction'] != 1:
                        return False
        else:
            return False
        return True

    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])
