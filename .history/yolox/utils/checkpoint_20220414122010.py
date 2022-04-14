#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name="", ap50_95=0, epoch=0 ,save_best_in_name=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        if save_best_in_name:
            best_filename = os.path.join(save_dir, 'epoch'+str(int(epoch)+1)+'_'+str(ap50_95)+"_best_ckpt.pth")
        else:
            best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)