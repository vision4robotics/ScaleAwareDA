# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck, Projection, Predictor
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise
import numpy as np
import cv2


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS).cuda()

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        if cfg.ALIGN.ALIGN:
            self.align = get_neck(cfg.ALIGN.TYPE,
                                 **cfg.ALIGN.KWARGS)
        
        # build projection
        self.projection = Projection(256,2048)
        self.predictor = Predictor(256,2048)
            
        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        if cfg.ALIGN.ALIGN:
            zf = [self.align(zf[i]) for i in range(len(zf))]
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        if cfg.ALIGN.ALIGN:
            xf = [self.align(xf[i]) for i in range(len(xf))]
        features = self.xcorr_depthwise(xf[0],self.zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)

        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()
        
        
        # source domain
        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        if cfg.ALIGN.ALIGN:
            zf = [self.align(zf[i]) for i in range(len(zf))]
            xf = [self.align(xf[i]) for i in range(len(xf))]


        # target domain
        rate = cfg.TRAIN.RATE
        b1,c1,h1,w1 = template.shape
        b2,c2,h2,w2 = search.shape
        template_copy = torch.zeros(template.shape)
        search_copy = torch.zeros(search.shape)
        
        for i in range(b1):
            tem_array = np.array(template[i].permute(1,2,0).cpu())
            tem_array_resize = cv2.resize(tem_array,(w1//rate, h1//rate))
            template_copy[i] = torch.from_numpy(cv2.resize(tem_array_resize,(w1, h1))).permute(2,0,1)

        for i in range(b2):
            sea_array = np.array(search[i].permute(1,2,0).cpu())
            sea_array_resize = cv2.resize(sea_array,(w2//rate, h2//rate))
            search_copy[i] = torch.from_numpy(cv2.resize(sea_array_resize,(w2, h2))).permute(2,0,1)
        template_copy = template_copy.cuda()
        search_copy = search_copy.cuda()
        
        # get feature    
        zf_copy = self.backbone(template_copy)
        xf_copy = self.backbone(search_copy)
        if cfg.ADJUST.ADJUST:
            zf_copy = self.neck(zf_copy)
            xf_copy = self.neck(xf_copy)
        if cfg.ALIGN.ALIGN:
            zf_copy = [self.align(zf_copy[i]) for i in range(len(zf_copy))]
            xf_copy = [self.align(xf_copy[i]) for i in range(len(xf_copy))]

        features_copy = self.xcorr_depthwise(xf_copy[0],zf_copy[0])
        for i in range(len(xf_copy)-1):
            features_new_copy = self.xcorr_depthwise(xf_copy[i+1],zf_copy[i+1])
            features_copy = torch.cat([features_copy,features_new_copy],1)
        features_copy = self.down(features_copy)

        cls_copy, loc_copy, cen_copy = self.car_head(features_copy)
        locations_copy = compute_locations(cls_copy, cfg.TRACK.STRIDE)
        cls_copy = self.log_softmax(cls_copy)
        cls_loss_copy, loc_loss_copy, cen_loss_copy = self.loss_evaluator(
            locations_copy,
            cls_copy,
            loc_copy,
            cen_copy, label_cls, label_loc
        )
        
        # contrastive loss evaluation
        x_pro = []; xc_pro = []; x_pre = []; xc_pre = [];
        z_pro = []; zc_pro = []; z_pre = []; zc_pre = [];
        mse_loss = nn.MSELoss()
        for i in range(len(xf)):
            x_pro.append(self.projection(xf[i]).detach())
            xc_pro.append(self.projection(xf_copy[i]).detach())
            z_pro.append(self.projection(zf[i]).detach())
            zc_pro.append(self.projection(zf_copy[i]).detach())
            x_pre.append(self.predictor(x_pro[i]))
            xc_pre.append(self.predictor(xc_pro[i]))
            z_pre.append(self.predictor(z_pro[i]))
            zc_pre.append(self.predictor(zc_pro[i]))
            
            
        x_dis_loss = [ mse_loss(x_pre[i], xc_pro[i]) for i in range(len(xf)) ]
        x_dis_loss = torch.mean(torch.stack(x_dis_loss))
        xc_dis_loss = [ mse_loss(xc_pre[i], x_pro[i]) for i in range(len(xf)) ]
        xc_dis_loss = torch.mean(torch.stack(xc_dis_loss))
        z_dis_loss = [ mse_loss(z_pre[i], zc_pro[i]) for i in range(len(zf)) ]
        z_dis_loss = torch.mean(torch.stack(z_dis_loss))
        zc_dis_loss = [ mse_loss(zc_pre[i], z_pro[i]) for i in range(len(zf)) ]
        zc_dis_loss = torch.mean(torch.stack(zc_dis_loss))
        

        da_loss = (x_dis_loss + xc_dis_loss + z_dis_loss + zc_dis_loss) * 0.25
        
        
       
        cls_loss = cls_loss_copy
        loc_loss = loc_loss_copy
        cen_loss = cen_loss_copy
        
        
        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss + cfg.TRAIN.DA_WEIGHT * da_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        outputs['da_loss'] = da_loss
        return outputs
