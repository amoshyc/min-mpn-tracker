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
    'KITTI-17': {
        'name': 'KITTI-17',
        'fps': 10,
        'imgW': 1224,
        'imgH': 370,
        'num_frames': 145,
        'camera': 'static',
    },
    'ETH-Sunnyday': {
        'name': 'ETH-Sunnyday',
        'fps': 14,
        'imgW': 640,
        'imgH': 480,
        'num_frames': 354,
        'camera': 'moving',
    },
    'ETH-Bahnhof': {
        'name': 'ETH-Bahnhof',
        'fps': 14,
        'imgW': 640,
        'imgH': 480,
        'num_frames': 1000,
        'camera': 'moving',
    },
    'PETS09-S2L1': {
        'name': 'PETS09-S2L1',
        'fps': 7,
        'imgW': 768,
        'imgH': 576,
        'num_frames': 795,
        'camera': 'static',
    },
    'TUD-Stadtmitte': {
        'name': 'TUD-Stadtmitte',
        'fps': 25,
        'imgW': 640,
        'imgH': 480,
        'num_frames': 179,
        'camera': 'static',
    },
}


if __name__ == '__main__':
    TARGET_DIR = Path(f'data/')
    MOT15_ROOT = Path('/store_2/2DMOT15/train/')
    MPN15_ROOT = Path('~/mot_neural_solver/data/2DMOT2015/train/').expanduser()
    
    for name, info in tqdm(INFO.items()):
        dest_dir = TARGET_DIR / name
        dest_dir.mkdir(exist_ok=True, parents=True)

        info['src_dir'] = str(MOT15_ROOT / name)
        with open(dest_dir / 'info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        mpn_dir = MPN15_ROOT / (name + '-GT') / 'processed_data'
        with open(mpn_dir / 'det' / 'gt.pkl', 'rb') as f:
            df_gt = pickle.load(f)
        df_gt = df_gt.sort_values(by='detection_id')
        df_gt.drop(columns=df_gt.columns[6:], inplace=True)
        df_gt.columns = ['fid', 'tag', 'x', 'y', 'w', 'h']
        df_gt['s'] = 1.0
        df_gt.to_csv(dest_dir / 'groundtruth.csv', index=None)

        emb_dir = mpn_dir / 'embeddings' / 'gt'
        for fid, group in df_gt.groupby('fid'):
            roi = torch.load(emb_dir / 'resnet50_conv' / f'{fid}.pt')[:, 1:]
            emb = torch.load(emb_dir / 'resnet50_w_fc256' / f'{fid}.pt')[:, 1:]
            assert len(roi) == len(emb)
            torch.save(roi, dest_dir / f'gt.reid.roi.{fid:06d}')
            torch.save(emb, dest_dir / f'gt.reid.emb.{fid:06d}')