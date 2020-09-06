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
from torchvision.ops import roi_pool, box_iou
from torchvision.transforms import functional as tf
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection._utils import Matcher
from torchvision.models.detection.transform import resize_boxes


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

reidmodel = reid_model('weights/resnet50_market_cuhk_duke.tar-232')
reidmodel = reidmodel.to('cuda')
reidmodel.eval()


@torch.no_grad()
def extract_embeddings(df: pd.DataFrame, img_dir: Path):
    check = torch.zeros(len(df))
    result = dict()

    name = img_dir.parent.stem
    for fid, group in df.groupby('fid'):
        boxes = group[['x', 'y', 'w', 'h']].values
        boxes[:, 2:] += boxes[:, :2]
        boxes = boxes.round().astype(int)
        image = Image.open(img_dir / f'{fid:06d}.jpg')

        inp_b = []
        for box in boxes:
            app = image.crop(box)
            app = app.resize((64, 128))
            app = tf.to_tensor(app)
            app = tf.normalize(app, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            inp_b.append(app)
        inp_b = torch.stack(inp_b)

        roi_b, emb_b = reidmodel(inp_b.to('cuda'))
        result[fid] = {'roi': roi_b.cpu(), 'emb': emb_b.cpu()}
        check[group.index.to_numpy()] = 1

    assert torch.all(check >= 1)
    return result


def match(df_gt: pd.DataFrame, df_pr: pd.DataFrame):
    gt_data = {fid: group for fid, group in df_gt.groupby('fid')}
    pr_data = {fid: group for fid, group in df_pr.groupby('fid')}

    missing_fids = set(gt_data.keys()) - set(pr_data.keys())
    if missing_fids != set():
        tqdm.write('No detections in frame {}'.format(missing_fids))

    results = []
    for fid in gt_data.keys():
        gt_group = gt_data[fid]
        gt_boxes = torch.from_numpy(gt_group[['x', 'y', 'w', 'h']].values).float()
        gt_boxes[:, 2:] += gt_boxes[:, :2]
        gt_tags = torch.from_numpy(gt_group['tag'].values).long()

        pr_group = pr_data.get(fid, None)
        if pr_group is None:
            continue
        pr_boxes = torch.from_numpy(pr_group[['x', 'y', 'w', 'h']].values).float()
        pr_boxes[:, 2:] += pr_boxes[:, :2]

        iou_matrix = box_iou(gt_boxes, pr_boxes)
        matches = Matcher(0.5, 0.5, False)(iou_matrix)
        matched_gt_boxes = torch.zeros(len(pr_group), 4)
        matched_gt_boxes[matches >= 0] = gt_boxes[matches[matches >= 0]]
        matched_gt_tags = -torch.ones(len(pr_group)).long()
        matched_gt_tags[matches >= 0] = gt_tags[matches[matches >= 0]]

        pr_boxes = pr_boxes.numpy()
        matched_gt_tags = matched_gt_tags.numpy()
        matched_gt_boxes = matched_gt_boxes.numpy()

        results.append(
            pd.DataFrame.from_dict(
                {
                    'fid': fid,
                    'tag': matched_gt_tags,
                    'x': pr_boxes[:, 0],
                    'y': pr_boxes[:, 1],
                    'w': pr_boxes[:, 2] - pr_boxes[:, 0],
                    'h': pr_boxes[:, 3] - pr_boxes[:, 1],
                    's': pr_group['s'].values,
                    'gx': matched_gt_boxes[:, 0],
                    'gy': matched_gt_boxes[:, 1],
                    'gw': matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0],
                    'gh': matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1],
                }
            )
        )

    return pd.concat(results, ignore_index=True)


if __name__ == '__main__':
    TARGET_DIR = Path(f'data/')
    MOT17_ROOT = Path('/store_2/MOT17/train/')
    MPN17_ROOT = Path('tracktor_detections/MOT17Labels/train/')

    for name, info in tqdm(INFO.items()):
        dest_dir = TARGET_DIR / name
        dest_dir.mkdir(exist_ok=True, parents=True)
        (dest_dir / 'gt.reid.roi').mkdir(exist_ok=True)
        (dest_dir / 'gt.reid.emb').mkdir(exist_ok=True)
        (dest_dir / 'det.reid.roi').mkdir(exist_ok=True)
        (dest_dir / 'det.reid.emb').mkdir(exist_ok=True)

        video_dir = MOT17_ROOT / name
        info['src_dir'] = str(video_dir)
        with open(dest_dir / 'info.json', 'w') as f:
            json.dump(info, f, indent=2)

        df_gt = pd.read_csv(video_dir / 'gt' / 'gt.txt', header=None)
        df_gt.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'ignore', 'class', 'vis']
        df_gt.to_csv(dest_dir / 'groundtruth.csv', index=None)
        df_gt['x'] -= 1
        df_gt['y'] -= 1
        reid_result = extract_embeddings(df_gt, video_dir / 'img1')
        for fid, reid in reid_result.items():
            torch.save(reid['roi'], dest_dir / 'gt.reid.roi' / f'{fid:06d}')
            torch.save(reid['emb'], dest_dir / 'gt.reid.emb' / f'{fid:06d}')

        df_det = pd.read_csv(
            MPN17_ROOT / name / 'det' / 'tracktor_prepr_det.txt', header=None
        )
        df_det.drop(columns=df_det.columns[6:], inplace=True)
        df_det.columns = ['fid', 'tag', 'x', 'y', 'w', 'h']
        df_det['x'] -= 1
        df_det['y'] -= 1
        df_det['s'] = 1.0
        df_det['tag'] = -1
        df_det = match(df_gt, df_det)
        df_det.to_csv(dest_dir / 'detections.csv', index=None)
        reid_result = extract_embeddings(df_det, video_dir / 'img1')
        for fid, reid in reid_result.items():
            torch.save(reid['roi'], dest_dir / 'det.reid.roi' / f'{fid:06d}')
            torch.save(reid['emb'], dest_dir / 'det.reid.emb' / f'{fid:06d}')
