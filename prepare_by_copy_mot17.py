from mpn_reid import reid_model

import json
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, List
from argparse import ArgumentParser
from torchvision.ops import roi_align, box_iou
from torchvision.transforms import functional as tf


INFO = {
    'MOT17-02-FRCNN': {
        'name': 'MOT17-02-FRCNN',
        'fps': 30,
        'imgW': 1920,
        'imgH': 1080,
        'num_frames': 600,
        'camera': 'static',
    },
    'MOT17-04-FRCNN': {
        'name': 'MOT17-04-FRCNN',
        'fps': 30,
        'imgW': 1920,
        'imgH': 1080,
        'num_frames': 1050,
        'camera': 'static',
    },
    'MOT17-05-FRCNN': {
        'name': 'MOT17-05-FRCNN',
        'fps': 14,
        'imgW': 640,
        'imgH': 480,
        'num_frames': 837,
        'camera': 'moving',
    },
    'MOT17-09-FRCNN': {
        'name': 'MOT17-09-FRCNN',
        'fps': 30,
        'imgW': 1920,
        'imgH': 1080,
        'num_frames': 525,
        'camera': 'static',
    },
    'MOT17-10-FRCNN': {
        'fps': 30,
        'name': 'MOT17-10-FRCNN',
        'imgW': 1920,
        'imgH': 1080,
        'num_frames': 654,
        'camera': 'moving',
    },
    'MOT17-11-FRCNN': {
        'name': 'MOT17-11-FRCNN',
        'fps': 30,
        'imgW': 1920,
        'imgH': 1080,
        'num_frames': 900,
        'camera': 'moving',
    },
    'MOT17-13-FRCNN': {
        'name': 'MOT17-13-FRCNN',
        'fps': 25,
        'imgW': 1920,
        'imgH': 1080,
        'num_frames': 750,
        'camera': 'moving',
    },
}

if __name__ == '__main__':
    TARGET_DIR = Path(f'data/')
    MOT17_ROOT = Path('/store_2/MOT17/train/')
    MPN17_ROOT = Path('~/mot_neural_solver/data/MOT17Labels/train/').expanduser()

    for name, info in tqdm(INFO.items()):
        dest_dir = TARGET_DIR / name
        dest_dir.mkdir(exist_ok=True, parents=True)
        (dest_dir / 'gt.reid.roi').mkdir(exist_ok=True)
        (dest_dir / 'gt.reid.emb').mkdir(exist_ok=True)
        (dest_dir / 'det.reid.roi').mkdir(exist_ok=True)
        (dest_dir / 'det.reid.emb').mkdir(exist_ok=True)

        info['src_dir'] = str(MOT17_ROOT / name)
        with open(dest_dir / 'info.json', 'w') as f:
            json.dump(info, f, indent=2)

        gt_dir = MPN17_ROOT / (name[:-6] + '-GT') / 'processed_data'
        with open(gt_dir / 'det' / 'gt.pkl', 'rb') as f:
            df_gt = pickle.load(f)
        df_gt = df_gt.sort_values(by='detection_id')
        df_gt.drop(columns=df_gt.columns[9:], inplace=True)
        df_gt.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', 'class', 'vis']
        df_gt.to_csv(dest_dir / 'groundtruth.csv', index=None)
        emb_dir = gt_dir / 'embeddings' / 'gt'
        for fid, group in df_gt.groupby('fid'):
            roi = torch.load(emb_dir / 'resnet50_conv' / f'{fid}.pt')[:, 1:]
            emb = torch.load(emb_dir / 'resnet50_w_fc256' / f'{fid}.pt')[:, 1:]
            assert len(roi) == len(emb)
            torch.save(roi, dest_dir / 'gt.reid.roi' / f'{fid:06d}')
            torch.save(emb, dest_dir / 'gt.reid.emb' / f'{fid:06d}')

        det_dir = MPN17_ROOT / name / 'processed_data'
        with open(det_dir / 'det' / 'tracktor_prepr_det.pkl', 'rb') as f:
            df_det = pickle.load(f)
        df_det = df_det.sort_values(by='detection_id')
        df_det.drop(columns=df_det.columns[6:], inplace=True)
        df_det.columns = ['fid', 'tag', 'x', 'y', 'w', 'h']
        df_det['s'] = 1.0
        df_det.to_csv(dest_dir / 'detections.csv', index=None)
        emb_dir = det_dir / 'embeddings' / 'tracktor_prepr_det'
        for fid, group in df_gt.groupby('fid'):
            roi = torch.load(emb_dir / 'resnet50_conv' / f'{fid}.pt')[:, 1:]
            emb = torch.load(emb_dir / 'resnet50_w_fc256' / f'{fid}.pt')[:, 1:]
            assert len(roi) == len(emb)
            torch.save(roi, dest_dir / 'det.reid.roi' / f'{fid:06d}')
            torch.save(emb, dest_dir / 'det.reid.emb' / f'{fid:06d}')