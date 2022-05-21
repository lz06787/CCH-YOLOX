from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (force_fp32, ga_loc_target, multi_apply,
                        multiclass_nms, PointGenerator, bga_target)
from mmdet.ops import DeformConv, MaskedConv2d, ConvModule
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob


class FeatureAdaption(nn.Module):
    """Feature Adaption Module.
    Feature Adaption Module is implemented based on DCN v1.
    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
        use_grid_bbox (bool): predict anchor by width and height
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
    """

    # deformable group?
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=1,
                 use_grid_bbox=False,
                 gradient_mul=0.1):
        super(FeatureAdaption, self).__init__()
        self.use_grid_bbox = use_grid_bbox
        self.gradient_mul = gradient_mul
        self.dcn_kernel = kernel_size
        self.dcn_pad = (self.dcn_kernel - 1) // 2
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        if self.use_grid_bbox:
            # use w,h to predict dcn offset
            offset_channels = kernel_size * kernel_size * 2
            self.conv_offset = nn.Conv2d(
                2, deformable_groups * offset_channels, 1, bias=False)
        else:
            dcn_base = np.arange(-self.dcn_pad,
                                 self.dcn_pad + 1).astype(np.float64)
            dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
            dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
            dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
                (-1))
            self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=self.dcn_kernel,
            padding=self.dcn_pad,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        if self.use_grid_bbox:
            normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape_pred):
        if self.use_grid_bbox:
            dcn_offset = self.conv_offset(shape_pred.detach())
        else:
            dcn_base_offset = self.dcn_base_offset.type_as(x)
            shape_pred = shape_pred * self.gradient_mul + \
                         shape_pred.detach() * (1 - self.gradient_mul)
            dcn_offset = shape_pred - dcn_base_offset
        x = self.relu(self.conv_adaption(x, dcn_offset))
        return x


@HEADS.register_module
class BetterGuidedAnchorHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=1,
                 num_points=9,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 gradient_mul=0.1,
                 deformable_groups=1,
                 loc_filter_thr=0.01,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_loc=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.5),
                 loss_shape=dict(
                     type='BoundedIoULoss', beta=0.2, loss_weight=0.5),
                 loss_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_bbox=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
                 use_grid_points=False,
                 use_bbox_refine=False,
                 transform_method='moment',
                 moment_mul=0.01):
        super(BetterGuidedAnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        self.point_strides = point_strides
        self.point_base_scale = point_base_scale
        self.gradient_mul = gradient_mul
        self.deformable_groups = deformable_groups
        self.loc_filter_thr = loc_filter_thr
        self.anchor_base_size = list(point_strides)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_grid_points = use_grid_points
        self.use_bbox_refine = use_bbox_refine
        self.transform_method = transform_method
        self.moment_mul = moment_mul
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self.cls_focal_loss = loss_cls['type'] in ['FocalLoss']

        # build losses
        self.loss_loc = build_loss(loss_loc)
        self.loss_shape = build_loss(loss_shape)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        # dcn
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        self._init_layers()

    def _init_layers(self):
        self.relu = self.relu = nn.ReLU(inplace=True)
        self.feat_refine_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.feat_refine_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        # use_grid_points: w,h   else 2 * numpoints
        pts_out_dim_init = 2 if self.use_grid_points else 2 * self.num_points
        pts_out_dim_refine = 4 if self.use_bbox_refine else 2 * self.num_points
        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_pts_init = nn.Conv2d(self.feat_channels, pts_out_dim_init, 1)
        self.feature_adaption = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)
        self.conv_cls = MaskedConv2d(self.feat_channels,
                                     self.cls_out_channels,
                                     1)
        self.conv_pts_refine = MaskedConv2d(self.feat_channels,
                                            pts_out_dim_refine,
                                            1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_pts_refine, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_pts_init, std=0.01)

        self.feature_adaption.init_weights()

    def forward_single(self, x):

        # 如果anchor处使用bbox
        if self.use_grid_points:
            # 如果use_grid_points为True，预测w,h
            pass
        else:
            points_init = 0

        # refine the feature map
        for feat_refine_conv in self.feat_refine_convs:
            x = feat_refine_conv(x)

        loc_pred = self.conv_loc(x)
        pts_pred_init = self.conv_pts_init(x)
        if self.use_grid_points:
            pass
        else:
            pts_pred_init = pts_pred_init + points_init
        x = self.feature_adaption(x, pts_pred_init)
        # masked conv is only used during inference for speed-up
        # Question: where is self.training
        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.conv_cls(x, mask)
        pts_pred_refine = self.conv_pts_refine(x, mask)
        if self.use_grid_points and self.use_bbox_refine:
            pass
        elif self.use_bbox_refine:
            pass
        else:
            pts_pred_refine = pts_pred_refine + pts_pred_init.detach()
        return loc_pred, pts_pred_init, cls_score, pts_pred_refine

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    # point原图座标和stride
    def get_points(self, featmap_sizes, img_metas):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points. Only used in MaxIOUAssigner.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale,
                                           scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat(
                    [point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate.
        """
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                # 将center的座标重复num_points次，之后用来计算points的座标
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                # two dimension,第一维为所有位置，第二维为num_points的偏移
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                # 求得各点的x,y的偏移
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                # 把xy_pts_shift转换成第一维是所有位置,第二维每两个数字对应一个点
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    # need coordinate value of points
    def points2bbox(self, pts, y_first=True):
        """
        Converting the points set into bounding box.
        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                          ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                          ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                    self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                dim=1)
        else:
            raise NotImplementedError
        return bbox

    # single layer(maybe more than one image)
    def loss_single(self, cls_score, pts_coordinate_refine, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        # gt bbox and predict bbox
        pts_coordinate_refine = pts_coordinate_refine.reshape(-1, 2 * self.num_points)
        bbox_pred = self.points2bbox(pts_coordinate_refine, y_first=False)
        # use SmoothL1Loss to compute the loss between gt bbox and prediction bbox
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss_loc_single(self, loc_pred, loc_target, loc_weight, loc_avg_factor,
                        cfg):
        loss_loc = self.loss_loc(
            loc_pred.reshape(-1, 1),
            loc_target.reshape(-1, 1).long(),
            loc_weight.reshape(-1, 1),
            avg_factor=loc_avg_factor)
        return loss_loc

    def loss_shape_single(self, pts_coordinate_init, bbox_gts, anchor_weights,
                          anchor_total_num):
        if self.use_grid_points:
            # TODO: add case of use grid points
            pass
        else:
            bbox_gts = bbox_gts.contiguous().view(-1, 4)
            anchor_weights = anchor_weights.contiguous().view(-1, 4)
            # filter out negative samples to speed-up weighted_bounded_iou_loss
            inds = torch.nonzero(anchor_weights[:, 0] > 0).squeeze(1)
            bbox_gts_ = bbox_gts[inds]
            anchor_weights_ = anchor_weights[inds]
            pts_coordinate_init = pts_coordinate_init.contiguous().view(-1, 2 * self.num_points)
            pts_coordinate_init = pts_coordinate_init[inds]
            if pts_coordinate_init.size()[0] == 0:
                pred_anchors_ = torch.empty([0, 4],dtype=torch.float).type_as(pts_coordinate_init)
            else:
                pred_anchors_ = self.points2bbox(pts_coordinate_init, y_first=False)
            assert pred_anchors_.size()[0] == anchor_weights_.size()[0], \
                'size unequal in loss shape single'
        loss_shape = self.loss_shape(
            pred_anchors_,
            bbox_gts_,
            anchor_weights_,
            avg_factor=anchor_total_num)
        return loss_shape

    def loss(self,
             loc_preds,
             pts_preds_init,
             cls_scores,
             pts_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        # print("loss function, len(pts_preds_init):{}".format(len(pts_preds_init)))
        # for lvl_idx in range(len(pts_preds_init)):
        #     print("loss function, pts_preds_init[{}]:{}".format(lvl_idx,
        #                                                         pts_preds_refine[lvl_idx].size()))
        featmap_sizes = [featmap.size()[-2:] for featmap in loc_preds]
        assert len(featmap_sizes) == len(self.point_generators)

        device = cls_scores[0].device
        label_channels = 1

        # target of location in anchor stage
        loc_targets, loc_weights, loc_avg_factor = ga_loc_target(
            gt_bboxes,
            featmap_sizes,
            self.point_base_scale,
            self.point_strides,
            center_ratio=cfg.center_ratio,
            ignore_ratio=cfg.ignore_ratio)

        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        if self.use_bbox_refine:
            pass
        else:
            pts_coordinate_preds_refine = self.offset_to_pts(center_list,
                                                             pts_preds_refine)

        # target of initial points
        guided_anchors_list, _ = self.get_anchors(
            featmap_sizes, pts_coordinate_preds_init, loc_preds, img_metas, device=device)
        init_sampling = False if not hasattr(cfg.init, 'sampler') else True
        if cfg.init.assigner['type'] == 'PointAssigner':
            # Assign target for center list
            candidate_list = center_list
        else:
            # transform center list to bbox list and
            #   assign target for bbox list
            bbox_list = self.centers_to_bboxes(center_list)
            candidate_list = bbox_list

        init_target = bga_target(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            cfg.init,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=init_sampling)
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init) = init_target
        anchor_total_num = (
            num_total_pos_init +
            num_total_neg_init if init_sampling else num_total_pos_init)

        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        refine_sampling = False if self.cls_focal_loss else True
        refine_target = bga_target(
            guided_anchors_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            cfg.refine,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=refine_sampling)
        (labels_list, label_weights_list, bbox_gt_list_refine,
         candidate_list_refine, bbox_weights_list_refine, num_total_pos_refine,
         num_total_neg_refine) = refine_target
        bbox_total_num = (
            num_total_pos_refine if self.cls_focal_loss else
                                num_total_pos_refine + num_total_neg_refine)

        # compute loss
        # get classification and bbox regression losses
        # num_level_proposals = [points.size(0) for points in center_list[0]]
        # pts_coordinate_preds_refine = self.levels_to_imgs(pts_coordinate_preds_refine)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            pts_coordinate_preds_refine,
            labels_list,
            label_weights_list,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
            num_total_samples=bbox_total_num,
            cfg=cfg)

        # get anchor location loss
        losses_loc = []
        for i in range(len(loc_preds)):
            loss_loc = self.loss_loc_single(
                loc_preds[i],
                loc_targets[i],
                loc_weights[i],
                loc_avg_factor=loc_avg_factor,
                cfg=cfg)
            losses_loc.append(loss_loc)

        # get anchor shape loss
        losses_shape = []
        for i in range(len(pts_preds_init)):
            loss_shape = self.loss_shape_single(
                pts_coordinate_preds_init[i],
                bbox_gt_list_init[i],
                bbox_weights_list_init[i],
                anchor_total_num=anchor_total_num)
            losses_shape.append(loss_shape)

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_shape=losses_shape,
            loss_loc=losses_loc)

    def get_anchors(self,
                    featmap_sizes,
                    pts_coordinate_init,
                    loc_preds,
                    img_metas,
                    use_loc_filter=False,
                    device='cuda'):
        """Get guided anchors according to initial prediction points
        and location mask
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            pts_coordinate_init (list[tensor]): Multi-level shape predictions.
            loc_preds (list[tensor]): Multi-level location predictions.
            img_metas (list[dict]): Image meta info.
            use_loc_filter (bool): Use loc filter or not.
            device (torch.device | str): device for returned tensors
        Returns:
            tuple: guided anchors of each image, loc masks of each image
        """
        num_levels = len(featmap_sizes)

        # for each image, we compute multi level guided anchors
        guided_anchors_list = []
        loc_mask_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_guided_anchors = []
            multi_level_loc_mask = []
            for i in range(num_levels):
                shape_pred = pts_coordinate_init[i][img_id]
                loc_pred = loc_preds[i][img_id]
                # 根据每张图片，每层的loc_mask筛选出有效的位置，并且提取出anchor
                loc_pred = loc_pred.sigmoid().detach()
                if use_loc_filter:
                    loc_mask = loc_pred >= self.loc_filter_thr
                else:
                    loc_mask = loc_pred >= 0.0
                mask = loc_mask.permute(1, 2, 0).reshape(-1)
                # TODO: 预测长宽的情况
                if self.use_grid_points:
                    pass
                else:
                    guided_anchors = self.points2bbox(shape_pred, y_first=False)
                    guided_anchors = guided_anchors[mask]
                multi_level_guided_anchors.append(guided_anchors)
                multi_level_loc_mask.append(loc_mask)
            guided_anchors_list.append(multi_level_guided_anchors)
            loc_mask_list.append(multi_level_loc_mask)
        return guided_anchors_list, loc_mask_list

    def get_bboxes(self,
                   loc_preds,
                   pts_preds_init,
                   cls_scores,
                   pts_preds_refine,
                   img_metas,
                   cfg,
                   rescale=False):
        assert len(cls_scores) == len(pts_preds_refine) == len(
            pts_preds_init) == len(loc_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        if self.use_bbox_refine:
            pass
        else:
            pts_coordinate_preds_refine = self.offset_to_pts(center_list,
                                                             pts_preds_refine)

        guided_anchors_list, loc_masks_list = self.get_anchors(
            featmap_sizes,
            pts_coordinate_preds_init,
            loc_preds,
            img_metas,
            use_loc_filter=not self.training,
            device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            pts_coordinate_refine_list = [
                pts_coordinate_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
            loc_mask_list = [
                loc_masks_list[img_id][i].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list,
                                               pts_coordinate_refine_list,
                                               loc_mask_list,
                                               img_shape,
                                               scale_factor,
                                               cfg,
                                               rescale)
            result_list.append(proposals)
        return result_list

    # input multi-level data
    def get_bboxes_single(self,
                          cls_scores,
                          pts_coordinate_refine,
                          mlvl_masks,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale):
        assert len(cls_scores) == len(pts_coordinate_refine)
        mlvl_bboxes = []
        mlvl_scores = []
        # every single layer
        for cls_score, point_coordinate_refine, mask in zip(cls_scores,
                                                            pts_coordinate_refine,
                                                            mlvl_masks):
            assert cls_score.size()[-2] * cls_score.size()[-1] \
                   == point_coordinate_refine.size()[-2]
            # if no location is kept, end.
            if mask.sum() == 0:
                continue
            # reshape scores and mask
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            mask = mask.permute(1, 2, 0).reshape(-1)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            scores = scores[mask, :]
            point_coordinate_refine = point_coordinate_refine[mask, :]
            if scores.dim() == 0:
                scores = scores.unsqueeze(0)
                point_coordinate_refine = point_coordinate_refine.unsqueeze(0)
            # filter anchors, bbox_pred, scores w.r.t. scores
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                point_coordinate_refine = point_coordinate_refine[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.points2bbox(point_coordinate_refine, y_first=False)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        # multi class NMS
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels