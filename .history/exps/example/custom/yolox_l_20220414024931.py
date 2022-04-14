#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 10
        self.depth = 0.33
        self.width = 0.5
        self.warmup_epochs = 5
        self.max_epoch = 150
        self.exp_name = 'yolox_l_baseline_cocofull'

        self.eval_interval = 3

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = './datasets/coco'
        self.train_ann = "instances_trainval.json"
        self.val_ann = "instances_test.json"
        self.test_ann = "instances_test.json"
        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        # --------------  training config --------------------- #
        self.no_aug_epochs = 50
        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
