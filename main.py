import json
import random
import heapq
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data as Graph, Batch
from torch_sparse import coalesce
from torch_scatter import scatter_add, scatter_min, scatter_max
from sknetwork.clustering import Louvain
from sknetwork.utils import edgelist2adjacency
from scipy.sparse import csr_matrix, csgraph

from utils import MLP, drop_entries, drop_objects, perturb_boxes
from utils import nms_per_fid_tag, interpolate_gaps
from utils import cls_metrics, mot_metrics
from utils import greedy_surpress, mpn_surpress
from trainer import MOTTrackTrainer


class Video(torch.utils.data.Dataset):
    def __init__(self, video_dir, W: int = 15, mode: str = 'train'):
        super().__init__()
        assert mode in ['train', 'infer']

        self.video_dir = Path(video_dir)
        self.mode = mode
        self.W = W
        with open(self.video_dir / 'info.json') as f:
            self.info = json.load(f)

        csv_name = 'groundtruth.csv' if mode == 'train' else 'detections.csv'
        self.dets = pd.read_csv(self.video_dir / csv_name)
        self.data = {fid: group for fid, group in self.dets.groupby('fid')}

        target_fps = 9 if self.info['camera'] == 'moving' else 6
        if target_fps >= self.info['fps']:
            self.step = 1
        else:
            self.step = round(self.info['fps'] / target_fps)

        self.fmin = self.dets['fid'].min()
        self.fmax = self.dets['fid'].max()
        if self.mode == 'infer':
            self.used_fids = list(range(self.fmin, self.fmax + 1, self.step))
            if self.used_fids[-1] != self.fmax:
                self.used_fids.append(self.fmax)
            self.used_fids = np.int32(self.used_fids)
        else:
            self.used_fids = list(range(self.fmin, self.fmax - W * self.step + 1))

    def __len__(self):
        if self.mode == 'infer':
            return len(self.used_fids) - self.W + 1
        else:
            return len(self.used_fids)

    def __getitem__(self, idx):
        if self.mode == 'infer':
            window_fids = self.used_fids[idx : idx + self.W]
            window = pd.concat([self.data[fid] for fid in window_fids])
            embs = torch.cat(
                [
                    torch.load(self.video_dir / 'det.reid.emb' / f'{fid:06d}')
                    for fid in window_fids
                ]
            )
        else:
            fid_s = self.used_fids[idx]
            if np.random.random() > 0.5:
                step = round(self.step * np.random.uniform(0.5, 1.5))
            else:
                step = self.step
            fid_t = min(fid_s + step * self.W, self.fmax)
            window_fids = np.linspace(fid_s, fid_t, self.W)
            window_fids = window_fids.round().astype(int)
            window = pd.concat([self.data[fid] for fid in window_fids])
            embs = torch.cat(
                [
                    torch.load(self.video_dir / 'gt.reid.emb' / f'{fid:06d}')
                    for fid in window_fids
                ]
            )

            window['wid'] = range(len(window))
            if 'class' in window and 'vis' in window:
                window = window[window['class'] == 1]
                window = window[window['vis'] >= 0.2]
            window = drop_objects(window)
            window = drop_entries(window)
            window = perturb_boxes(window)
            embs = embs[window['wid'].values]

            assert len(window) > 0, window

        graph = Graph(
            vert_fid=torch.from_numpy(window['fid'].values).float(),
            vert_sec=torch.from_numpy(window['fid'].values).float() / self.info['fps'],
            vert_box=torch.from_numpy(window[['x', 'y', 'w', 'h']].values).float(),
            vert_pid=torch.from_numpy(window.index.to_numpy()),
            vert_tag=torch.from_numpy(window['tag'].values).long(),
            vert_emb=embs,
        )
        graph.num_nodes = len(window)

        return graph

    def load_gt(self):
        return pd.read_csv(self.video_dir / 'groundtruth.csv')


class MPNAssoc(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

        self.vert_encoder = MLP([256, 128, 32])
        self.edge_encoder = MLP([6, 18, 18, 16])

        self.edge_mlp = MLP([96 + 32 + 32, 80, 16])
        self.vert_mlp = MLP([64, 32, 32])
        self.flow_in_mlp = MLP([48, 56, 32])
        self.flow_out_mlp = MLP([48, 56, 32])

        self.edge_classifier = MLP([16, 16, 1], last_act=None)

    def forward(self, graph_b):
        vert_fid = graph_b.vert_fid
        vert_sec = graph_b.vert_sec
        vert_box = graph_b.vert_box
        vert_emb = graph_b.vert_emb
        us, vs = graph_b.edge_index
        V = graph_b.num_nodes

        boxX, boxY = vert_box[:, 0], vert_box[:, 1]
        boxW, boxH = vert_box[:, 2], vert_box[:, 3]
        boxX += boxW / 2
        boxY += boxH

        s = torch.min(us, vs)
        t = torch.max(us, vs)
        edge_attr = torch.stack(
            [
                (vert_sec[t] - vert_sec[s]).float(),
                2 * (boxX[t] - boxX[s]) / (boxH[t] + boxH[s]),
                2 * (boxY[t] - boxY[s]) / (boxH[t] + boxH[s]),
                torch.log(boxH[t]) - torch.log(boxH[s]),
                torch.log(boxW[t]) - torch.log(boxW[s]),
                (vert_emb[t] - vert_emb[s]).norm(dim=1),
            ],
            dim=1,
        )

        vert_feat = self.vert_encoder(vert_emb)
        edge_feat = self.edge_encoder(edge_attr)
        initial_vert_feat = vert_feat
        initial_edge_feat = edge_feat

        edge_preds = []
        for t in range(self.T):
            edge_feat = self.edge_mlp(
                torch.cat(
                    [
                        vert_feat[us],
                        vert_feat[vs],
                        edge_feat,
                        initial_edge_feat,
                        initial_vert_feat[us],
                        initial_vert_feat[vs],
                    ],
                    dim=1,
                )
            )
            flow_in = self.flow_in_mlp(torch.cat([vert_feat[us], edge_feat], dim=1))
            flow_out = self.flow_out_mlp(torch.cat([vert_feat[vs], edge_feat], dim=1))
            vert_feat = self.vert_mlp(
                torch.cat(
                    [
                        scatter_add(flow_in, vs, dim=0, dim_size=V),
                        scatter_add(flow_out, us, dim=0, dim_size=V),
                    ],
                    dim=1,
                )
            )

            edge_preds.append(self.edge_classifier(edge_feat).flatten())

        return edge_preds


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.K = 50
        self.assoc = MPNAssoc(12)

    def forward(self, graphs: List[Graph]):
        for i, graph in enumerate(graphs):
            mask_time = graph.vert_fid.view(-1, 1) != graph.vert_fid.view(1, -1)
            dist = torch.cdist(graph.vert_emb, graph.vert_emb)
            dist[~mask_time] = float('inf')
            rank = dist.argsort(dim=1).argsort(dim=1)
            mask_knn = (rank < self.K) * (rank.t() < self.K)
            mask = mask_time & mask_knn
            assert torch.any(mask), graph
            us, vs = torch.where(mask)
            graphs[i].edge_index = torch.stack([us, vs], dim=0)

        graph_b = Batch.from_data_list(graphs)
        edge_preds = self.assoc(graph_b)

        V = graph_b.num_nodes
        device = graph_b.vert_fid.device
        tag_matrix = torch.zeros(V, V, device=device)
        for tag in torch.unique(graph_b.vert_tag):
            if tag == -1:
                continue
            idxs = torch.where(graph_b.vert_tag == tag)[0]
            tag_matrix[idxs[:-1], idxs[1:]] = 1
            tag_matrix[idxs[1:], idxs[:-1]] = 1
        edge_true = tag_matrix[graph_b.edge_index[0], graph_b.edge_index[1]]

        n_positive = edge_true.sum()
        n_negative = len(edge_true) - n_positive
        pos_weight = (
            n_negative / n_positive if n_positive > 0 else None
        )
        loss_edge = sum(
            [
                F.binary_cross_entropy_with_logits(
                    edge_pred, edge_true, pos_weight=pos_weight
                )
                for edge_pred in edge_preds
            ]
        )

        edge_preds = [torch.sigmoid(edge_pred.detach()) for edge_pred in edge_preds]

        output_dict = {
            'vert_pid': graph_b.vert_pid,
            'edge_index': graph_b.edge_index,
            'edge_preds': edge_preds,
            'edge_true': edge_true,
            'pos_weight': pos_weight,
        }
        loss_dict = {
            'loss_total': loss_edge,
        }

        return output_dict, loss_dict


@torch.no_grad()
def track(model, video, use_gt=False):
    model.eval()
    loader = torch.utils.data.DataLoader(
        video, batch_size=2, shuffle=False, num_workers=1, collate_fn=lambda x: x,
    )

    edge_pids = []
    edge_preds = []
    for graphs in tqdm(iter(loader), desc=f'Track {video.info["name"]}', leave=False):
        graphs = [graph.to('cuda') for graph in graphs]
        output_dict, _ = model.forward(graphs)

        if use_gt:
            edge_pred = output_dict['edge_true']
        else:
            edge_pred = output_dict['edge_preds'][-1]

        vert_pid = output_dict['vert_pid']
        edge_true = output_dict['edge_true']
        edge_index = output_dict['edge_index']
        edge_pids.append(vert_pid[edge_index].cpu())
        edge_preds.append(edge_pred.cpu())

        del graphs
        del output_dict

    pred = video.dets.copy()
    pred = pred[['fid', 'x', 'y', 'w', 'h', 's']]

    # Averaging between same directed edges due to sliding window
    edge_pids = torch.cat(edge_pids, dim=1)
    edge_preds = torch.cat(edge_preds, dim=0)
    edge_pids, edge_preds = coalesce(
        edge_pids, edge_preds, len(pred), len(pred), op='mean'
    )

    # Undirected to directed edges
    edge_pids = torch.stack(
        [torch.min(edge_pids[0], edge_pids[1]), torch.max(edge_pids[0], edge_pids[1]),]
    )
    edge_pids, edge_preds = coalesce(
        edge_pids, edge_preds, len(pred), len(pred), op='mean'
    )

    # Keep positive edges
    pos_mask = edge_preds > 0.5
    edge_pids = edge_pids[:, pos_mask]
    edge_preds = edge_preds[pos_mask]

    if len(edge_pids) == 0:
        return pd.DataFrame(
            [{'fid': 1, 'x': -100, 'y': -100, 'w': 1, 'h': 1, 's': 0.0}]
        )

    # Surpress
    edge_pids = greedy_surpress(edge_pids, edge_preds, len(pred))
    edge_pids = edge_pids.numpy().T

    # DFS
    graph = csr_matrix(
        (np.ones(len(edge_pids)), (edge_pids[:, 0], edge_pids[:, 1])),
        shape=(len(pred), len(pred)),
    )
    n_cluster, labels = csgraph.connected_components(graph, directed=False)
    pred['tag'] = labels + 1

    # Louvain
    # louvain = Louvain(sort_clusters=False, resolution=0.1)
    # adj = edgelist2adjacency(edge_pids)
    # adj.resize(len(pred), len(pred))
    # labels = louvain.fit_transform(adj)
    # pred['tag'] = labels + 1

    pred = nms_per_fid_tag(pred)
    pred = pred.sort_values(by='s')
    pred = pred[~pred.duplicated(subset=['fid', 'tag'])]
    pred = pred.groupby('tag').filter(lambda group: len(group) > 1)
    pred['tag'], _ = pd.factorize(pred['tag'])

    pred = interpolate_gaps(pred)
    pred = pred[['fid', 'tag', 'x', 'y', 'w', 'h', 's']]
    pred = pred.sort_values(by=['tag', 'fid'])

    pred['x'] += 1
    pred['y'] += 1

    return pred


def batch_to_metrics(graphs, output_dict, loss_dict):
    metrics = loss_dict
    edge_pred = output_dict['edge_preds'][-1].detach() > 0.5
    edge_true = output_dict['edge_true'].detach() > 0.5
    metrics.update(
        {f'edge_cls_{k}': v for k, v in cls_metrics(edge_pred, edge_true).items()}
    )
    return metrics


if __name__ == '__main__':
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DATA_DIR = Path('data')
    train_videos = [
        Video(DATA_DIR / 'KITTI-17', W=15, mode='train'),
        Video(DATA_DIR / 'ETH-Sunnyday', W=15, mode='train'),
        Video(DATA_DIR / 'ETH-Bahnhof', W=15, mode='train'),
        Video(DATA_DIR / 'PETS09-S2L1', W=15, mode='train'),
        Video(DATA_DIR / 'TUD-Stadtmitte', W=15, mode='train'),
        Video(DATA_DIR / 'MOT17-04-FRCNN', W=15, mode='train'),
        Video(DATA_DIR / 'MOT17-11-FRCNN', W=15, mode='train'),
        Video(DATA_DIR / 'MOT17-05-FRCNN', W=15, mode='train'),
        Video(DATA_DIR / 'MOT17-09-FRCNN', W=15, mode='train'),
    ]
    valid_videos = [
        Video(DATA_DIR / 'MOT17-02-FRCNN', W=15, mode='infer'),
        Video(DATA_DIR / 'MOT17-10-FRCNN', W=15, mode='infer'),
        Video(DATA_DIR / 'MOT17-13-FRCNN', W=15, mode='infer'),
    ]

    # for video in train_videos:
    #     for graph in video:
    #         pass
    # input()

    model = Model()
    model = model.to('cuda')
    model.eval()

    curr_time = datetime.now().strftime('%b-%d %H:%M:%S')
    log_dir = Path('runs') / __file__[:-3] / curr_time
    log_dir.mkdir(parents=True)
    shutil.copy(__file__, log_dir)
    logger = SummaryWriter(log_dir)
    print('Log dir: ', log_dir)

    names = []
    df_preds = []
    df_trues = []
    for video in valid_videos:
        df_pred = track(model, video, use_gt=True)
        df_pred.to_csv(log_dir / f'{video.info["name"]}.txt', index=None, header=None)
        df_true = video.load_gt()
        names.append(video.info['name'])
        df_preds.append(df_pred)
        df_trues.append(df_true)
    print('Upperbound:')
    summary = mot_metrics(df_preds, df_trues, names)
    for name, metrics in summary.items():
        for k, v in metrics.items():
            logger.add_scalar(f'Upperbound-{name}/{k}', v, 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 7, 0.5)
    trainer = MOTTrackTrainer(
        model,
        optimizer,
        scheduler,
        log_dir,
        track_fn=track,
        train_log_fn=batch_to_metrics,
        valid_log_fn=batch_to_metrics,
        n_epoch=5,
    )
    trainer.fit(train_videos, valid_videos)

