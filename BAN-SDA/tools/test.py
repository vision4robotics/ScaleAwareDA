from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from siamban.core.config import cfg
from siamban.models.model_builder import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.bbox import get_axis_aligned_bbox
from siamban.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory, datapath
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='bansda tracking')
parser.add_argument('--dataset', default='UAVTrack112', type=str,
                    help='datasets')#UAV123_20L,UAVDT
parser.add_argument('--datasetpath', default='', type=str,
                    help='the path of datasets')
parser.add_argument('--config', default='../experiments/bansda_r50_l234/config.yaml', type=str,
                    help='config file')
parser.add_argument('--snapshot', default='', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
                    help='eval one special video')
parser.add_argument('--vis', default='', action='store_true',
                    help='whether visualzie result') #true
parser.add_argument('--trackername', default='', type=str,
                    help='name of tracker')
args = parser.parse_args()


def main():
    
    
    # load config
    cfg.merge_from_file(args.config)

    # hp_search
    params = getattr(cfg.HP_SEARCH,args.dataset)
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)


    for dataset_name in args.dataset.split(','):
        # create dataset
        try:
            dataset_root = args.datasetpath + datapath[dataset_name]
        except:
            print('?')
        dataset = DatasetFactory.create_dataset(name=dataset_name,
                                            dataset_root=dataset_root,
                                            load_img=False)
        model_name = args.trackername

        # OPE tracking
        IDX = 0
        TOC = 0
        for v_idx, video in enumerate(dataset): 
          

            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img,hp)  
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results 
            model_path = os.path.join('results', dataset_name, model_name) #results
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(v_idx+1, video.name, toc, idx / toc))
            IDX += idx
            TOC += toc
        print('Total Time: {:5.1f}s Average Speed: {:3.1f}fps'.format(TOC, IDX / TOC))
        fps_path = os.path.join('results', dataset_name, '{}.txt'.format(model_name))
        with open(fps_path, 'w') as f:
            f.write('Time:{:5.1f},Speed:{:3.1f}'.format(TOC, IDX / TOC))


if __name__ == '__main__':
    main()
