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


class MultiViewTotalCapture(JointsDataset):

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
            9: 'head',
            10: 'lsho',
            11: 'lelb',
            12: 'lwri',
            13: 'rsho',
            14: 'relb',
            15: 'rwri'
        }

        if is_train:
            anno_file = osp.join(self.root, 'totalcapture', 'annot',
                                 'totalcapture_{}_new_17_pseudo.pkl'.format(image_set))
        else:
            anno_file = osp.join(self.root, 'totalcapture', 'annot',
                                 'totalcapture_{}_new_17.pkl'.format(image_set))

        self.db = self.load_db(anno_file)

        # self.u2a_mapping = super().get_mapping()
        # super().do_mapping()

        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)

    def index_to_action_names(self):
        return {
            2: 'Direction',
            3: 'Discuss',
            4: 'Eating',
            5: 'Greet',
            6: 'Phone',
            7: 'Photo',
            8: 'Pose',
            9: 'Purchase',
            10: 'Sitting',
            11: 'SittingDown',
            12: 'Smoke',
            13: 'Wait',
            14: 'WalkDog',
            15: 'Walk',
            16: 'WalkTwo'
        }

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset_all_view = pickle.load(f)
        dataset = []
        for item in dataset_all_view:
            if item['camera_id'] % 2 == 0:
                item['camera_id'] = int(item['camera_id']/2)
                dataset.append(item)
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
            filtered_grouping = filtered_grouping[::10]
            # filtered_grouping = filtered_grouping[::640]

        # if self.is_train:
        #     filtered_grouping = filtered_grouping[:1]
        # else:
        #     # pass
        #     filtered_grouping = filtered_grouping[46635:46636]
        #     # filtered_grouping = filtered_grouping[46636:46637]
        #     # filtered_grouping = filtered_grouping[:1]



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

    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()

        headsize = self.image_size[0] / 10.0
        threshold = 0.5

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))

        gt = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])
        gt = np.array(gt)
        pred = pred[:, su, :2]

        distance = np.sqrt(np.sum((gt - pred) ** 2, axis=2))
        detected = (distance <= headsize * threshold)

        joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]
        return name_values, np.mean(joint_detection_rate)
