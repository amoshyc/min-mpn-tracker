import torch
from torch import nn
from torchvision.ops.boxes import batched_nms

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import motmetrics

from torch_scatter import scatter_add
from collections import defaultdict


def perturb_boxes(window: pd.DataFrame, min_iou: float = 0.8) -> pd.DataFrame:
    uniform = np.random.uniform
    x1, x2 = window['x'].values, window['x'].values + window['w'].values
    y1, y2 = window['y'].values, window['y'].values + window['h'].values
    eps = min((1 - np.sqrt(min_iou)) / 2, 0.5 * (1  / np.sqrt(min_iou) - 1))
    noise = np.random.uniform(-eps, eps, (len(window), 4))
    noise = noise * window[['w', 'w', 'h', 'h']].values
    x1, x2 = x1 + noise[:, 0], x2 + noise[:, 1]
    y1, y2 = y1 + noise[:, 2], y2 + noise[:, 3]
    window['x'], window['w'] = x1, (x2 - x1)
    window['y'], window['h'] = y1, (y2 - y1)
    return window


def drop_entries(
    window: pd.DataFrame, prob: Tuple[float, float] = (0.0, 0.3)
) -> pd.DataFrame:
    prob = np.random.uniform(prob[0], prob[1])
    keep_mask = np.random.uniform(size=len(window)) >= prob
    if keep_mask.sum() >= 30:
        window = window[keep_mask]
    return window


def drop_objects(
    window: pd.DataFrame, prob: Tuple[float, float] = (0.0, 0.15)
) -> pd.DataFrame:
    all_tags = window['tag'].unique()
    prob = np.random.uniform(prob[0], prob[1])
    sampled_tags = all_tags[np.random.uniform(size=len(all_tags)) >= prob]
    keep_mask = window['tag'].isin(sampled_tags)
    if keep_mask.sum() >= 30:
        window = window[keep_mask]
    return window


def interpolate_gaps(pred: pd.DataFrame) -> pd.DataFrame:
    gaps = []
    for tag, tracklet in pred.groupby('tag'):
        index_no_gap = range(tracklet['fid'].min(), tracklet['fid'].max() + 1)
        tracklet = tracklet.sort_values(by=['fid', 'tag'])
        tracklet = tracklet.copy().set_index('fid')
        boxes = tracklet[['x', 'y', 'w', 'h', 's']].reindex(index=index_no_gap)
        mask = pd.isna(boxes['x'])
        boxes = boxes.interpolate()
        gap = boxes[mask].copy()
        gap['tag'] = tag
        gap['fid'] = gap.index.astype(int)
        gaps.append(gap)
    gaps = pd.concat(gaps, ignore_index=True)
    pred = pd.concat([pred, gaps])
    return pred


def nms_per_fid_tag(pred: pd.DataFrame) -> pd.DataFrame:
    boxes = torch.from_numpy(pred[['x', 'y', 'w', 'h']].values).float()
    boxes[:, 2:] += boxes[:, :2]
    scores = torch.from_numpy(pred['s'].values).float()
    categories, _ = pd.factorize(
        pd._lib.fast_zip([pred['fid'].values, pred['tag'].values])
    )
    categories = torch.from_numpy(categories).long()
    keep = batched_nms(boxes, scores, categories, 0.5).numpy()
    return pred.iloc[keep].copy()


def cls_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict:
    y_pred, y_true = y_pred.long(), y_true.long()
    tp = ((y_pred == 1) & (y_true == 1)).long().sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).long().sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).long().sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).long().sum().item()
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-15)
    prec = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)
    f1 = 2 * prec * recall / (prec + recall + 1e-15)
    return {'acc': acc, 'prec': prec, 'recall': recall, 'f1': f1}


# def mot_metrics(
#     df_pred: pd.DataFrame, df_true: pd.DataFrame, clean: bool = False
# ) -> Dict:
#     accum = motmetrics.mot.MOTAccumulator()
#     n_remove = 0

#     data_pred = {fid: group for fid, group in df_pred.groupby('fid')}
#     data_true = {fid: group for fid, group in df_true.groupby('fid')}
#     for fid in data_true.keys():
#         group_true = data_true[fid]
#         if fid not in data_pred:
#             continue  # Need fix

#         group_pred = data_pred[fid]

#         if clean:
#             rr, cc = motmetrics.lap.linear_sum_assignment(
#                 motmetrics.distances.iou_matrix(
#                     group_true[['x', 'y', 'w', 'h']].values,
#                     group_pred[['x', 'y', 'w', 'h']].values,
#                     0.5,
#                 )
#             )
#             distractor_mask = group_true.iloc[rr]['class'].isin([2, 6, 7, 8, 12])
#             if distractor_mask.sum() > 0:
#                 group_pred = group_pred.iloc[cc[~distractor_mask]]
#                 n_remove += distractor_mask.sum().item()

#         group_true = group_true[group_true['class'] == 1]
#         dist = motmetrics.distances.iou_matrix(
#             group_true[['x', 'y', 'w', 'h']].values,
#             group_pred[['x', 'y', 'w', 'h']].values,
#             0.5,
#         )
#         accum.update(
#             group_true['tag'].values, group_pred['tag'].values, dist, frameid=fid
#         )

#     metrics = motmetrics.metrics.create()
#     METRICS = [
#         'idf1',
#         'mota',
#         'mostly_tracked',
#         'mostly_lost',
#         'partially_tracked',
#         'num_false_positives',
#         'num_misses',
#         'num_switches',
#         'precision',
#         'recall',
#     ]
#     summary = metrics.compute(accum, metrics=METRICS)
#     summary = summary.to_dict('records')[0]
#     summary.update({'n_remove': n_remove})

#     return summary


def mot_metrics(
    df_preds: List[pd.DataFrame], df_trues: List[pd.DataFrame], names: List[str]
):
    accumulates = []
    for df_pred, df_true in zip(df_preds, df_trues):
        df_true = df_true[df_true['class'] == 1]
        df_pred['x'] -= 1
        df_pred['y'] -= 1
        df_pred = df_pred.set_index(['fid', 'tag'])
        df_true = df_true.set_index(['fid', 'tag'])
        accumulates.append(
            motmetrics.utils.compare_to_groundtruth(
                df_true, df_pred, distfields=['x', 'y', 'w', 'h']
            )
        )
    metrics = motmetrics.metrics.create()
    summary = metrics.compute_many(
        accumulates,
        names=names,
        metrics=motmetrics.metrics.motchallenge_metrics
        + ['num_objects', 'idtp', 'idfn', 'idfp', 'num_predictions'],
        generate_overall=True,
    )
    print(
        motmetrics.io.render_summary(
            summary,
            formatters=metrics.formatters,
            namemap=motmetrics.io.motchallenge_metric_names,
        )
    )
    return summary.to_dict('index')


# def compute_mot_metrics(gt_path, out_mot_files_path, seqs, print_results=True):
#     """
#     The following code is adapted from
#     https://github.com/cheind/py-motmetrics/blob/develop/motmetrics/apps/eval_motchallenge.py
#     It computes all MOT metrics from a set of output tracking files in MOTChallenge format
#     Args:
#         gt_path: path where MOT ground truth files are stored. Each gt file must be stored as
#         <SEQ NAME>/gt/gt.txt
#         out_mot_files_path: path where output files are stored. Each file must be named <SEQ NAME>.txt
#         seqs: Names of sequences to be evalluated

#     Returns:
#         Individual and overall MOTmetrics for all sequeces
#     """
#     mm = motmetrics

#     def _compare_dataframes(gts, ts):
#         """Builds accumulator for each sequence."""
#         accs = []
#         names = []
#         for k, tsacc in ts.items():
#             if k in gts:
#                 # print(gts[k].head(10))
#                 # print(tsacc.head(10))
#                 accs.append(
#                     mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5)
#                 )
#                 names.append(k)

#         return accs, names

#     gtfiles = [os.path.join(gt_path, i, 'gt/gt.txt') for i in seqs]
#     tsfiles = [os.path.join(out_mot_files_path, '%s.txt' % i) for i in seqs]

#     gt = dict(
#         [
#             (Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1))
#             for f in gtfiles
#         ]
#     )
#     ts = dict(
#         [
#             (os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D'))
#             for f in tsfiles
#         ]
#     )

#     mh = mm.metrics.create()
#     accs, names = _compare_dataframes(gt, ts)

#     # We will need additional metrics to compute IDF1, etc. from different splits inf CrossValidationEvaluator
#     summary = mh.compute_many(
#         accs,
#         names=names,
#         metrics=mm.metrics.motchallenge_metrics
#         + ['num_objects', 'idtp', 'idfn', 'idfp', 'num_predictions'],
#         generate_overall=True,
#     )
#     if print_results:
#         print(
#             mm.io.render_summary(
#                 summary,
#                 formatters=mh.formatters,
#                 namemap=mm.io.motchallenge_metric_names,
#             )
#         )

#     return summary.to_dict('index')


def compute_constr_sat_rate(edge_index, edge_pred, V, undirected=True):
    if undirected:
        edge_index, _ = edge_index.t().sort(dim=1)
        edge_index = edge_index.t()
        div_factor = 2
    else:
        div_factor = 1

    flow_out = scatter_add(edge_pred, edge_index[0], dim_size=V) / div_factor
    flow_in = scatter_add(edge_pred, edge_index[1], dim_size=V) / div_factor

    violated_out = (flow_out > 1).sum().item()
    violated_in = (flow_in > 1).sum().item()
    violated_num = violated_out + violated_in
    constrnt_num = len(edge_index[0].unique()) + len(edge_index[1].unique())
    sat_rate = 1 - violated_num / constrnt_num
    return sat_rate, flow_in, flow_out


def mpn_surpress(edge_index, edge_pred, V):
    round_pred = (edge_pred > 0.5).float()

    sat_rate, flow_in, flow_out = compute_constr_sat_rate(
        edge_index, round_pred, V, undirected=False
    )

    device = edge_index.device
    vert_names = torch.arange(V).to(device)
    type_in = torch.zeros(V).to(device)
    type_out = torch.ones(V).to(device)

    flow_in_info = torch.stack((vert_names, type_in)).t().float()  # [V, 2]
    flow_out_info = torch.stack((vert_names, type_out)).t().float()  # [V, 2]
    all_constr = torch.cat((flow_in_info, flow_out_info))  # [2V, 2]
    violated_mask = torch.cat(((flow_in > 1), (flow_out > 1)))  # [2V]

    violated_constr = all_constr[violated_mask]  # [N, 2]
    violated_constr = violated_constr[
        torch.argsort(violated_constr[:, 1], descending=True)
    ]

    for constr in violated_constr:
        vert_name, violated_type = constr

        mask = torch.zeros(V).bool()
        mask[vert_name.long()] = True
        if violated_type == 0:  # flow in violation
            mask = mask[edge_index[1]]  # [E]
        else:  # flow out violation
            mask = mask[edge_index[0]]  # [E]
        mask_index = torch.where(mask)[0]

        if round_pred[mask_index].sum() > 1:
            j = max(mask_index, key=lambda idx: edge_pred[idx] * round_pred[idx])
            round_pred[mask] = 0
            round_pred[j] = 1

    assert scatter_add(round_pred, edge_index[1], dim_size=V).max() <= 1
    assert scatter_add(round_pred, edge_index[0], dim_size=V).max() <= 1

    return round_pred


def greedy_surpress(edge_index, edge_preds, V):
    # Constraint 1
    adj_futr = defaultdict(list)
    keep_mask = torch.ones(edge_index.size(1)).long()
    for i, ((u, v), s) in enumerate(zip(edge_index.t(), edge_preds)):
        u, v, s = u.item(), v.item(), s.item()
        adj_futr[u].append((v, s, i))
    for u, adjs in adj_futr.items():
        adjs = sorted(adjs, key=lambda x: x[1], reverse=True)
        for (v, s, i) in adjs[1:]:
            keep_mask[i] = 0
    edge_index = edge_index[:, keep_mask == 1]
    edge_preds = edge_preds[keep_mask == 1]

    # Constraint 2
    adj_past = defaultdict(list)
    keep_mask = torch.ones(edge_index.size(1)).long()
    for i, ((u, v), s) in enumerate(zip(edge_index.t(), edge_preds)):
        u, v, s = u.item(), v.item(), s.item()
        adj_past[v].append((u, s, i))
    for v, adjs in adj_past.items():
        adjs = sorted(adjs, key=lambda x: x[1], reverse=True)
        for (u, s, i) in adjs[1:]:
            keep_mask[i] = 0

    edge_index = edge_index[:, keep_mask == 1]
    edge_preds = edge_preds[keep_mask == 1]

    return edge_index


class MLP(nn.Sequential):
    def __init__(self, dims, bn=False, last_act=nn.ReLU):
        layers = []
        for i, dim in enumerate(dims[:-1]):
            layers.append(nn.Linear(dim, dims[i + 1]))
            if bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if i == len(dims) - 2:
                if last_act is not None:
                    layers.append(last_act())
            else:
                layers.append(nn.ReLU())
        super().__init__(*layers)


if __name__ == '__main__':
    edge_index = torch.tensor([[5, 8], [5, 9], [6, 9], [7, 11], [8, 10], [8, 11]]).t()
    edge_pred = torch.rand(6) + 0.5
    print(edge_pred)

    round_pred = mpn_surpress(edge_index, edge_pred, 12)

    for (u, v), s in zip(edge_index.t(), round_pred):
        if s == 1:
            print(u, v)

    print('-' * 10)
    keep_edges = greedy_surpress(edge_index, edge_pred, 12)
    for (u, v) in keep_edges.t():
        print(u, v)
