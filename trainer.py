import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Iterable, Dict
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from utils import mot_metrics


class MOTTrackTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        log_dir,
        track_fn=None,
        train_log_fn=None,
        valid_log_fn=None,
        n_epoch=10,
        log_interval=10,
    ):
        self.model = model.to('cuda')
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.track_fn = track_fn
        self.train_log_fn = train_log_fn
        self.valid_log_fn = valid_log_fn
        self.n_epoch = n_epoch
        self.log_interval = log_interval

        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = SummaryWriter(self.log_dir)

    def fit(self, train_videos: List[Dataset], valid_videos: List[Dataset]):
        train_loader = DataLoader(
            ConcatDataset(train_videos),
            batch_size=8,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda x: x,
        )

        self.step = 1
        for self.epoch in range(1, self.n_epoch + 1):
            epoch_dir = self.log_dir / f'{self.epoch:03d}'
            epoch_dir.mkdir(exist_ok=True)

            # Train
            self.model.train()
            for graphs in tqdm(iter(train_loader), desc=f'Train {self.epoch:03d}'):
                self.optimizer.zero_grad()
                graphs = [graph.to('cuda') for graph in graphs]
                output_dict, loss_dict = self.model(graphs)
                loss = loss_dict['loss_total']
                if torch.isnan(loss):
                    raise ValueError('loss is NaN!')
                loss.backward()
                self.optimizer.step()
                self.step += 1

                if self.step % self.log_interval == 0:
                    metrics = self.train_log_fn(graphs, output_dict, loss_dict)
                    for k, v in metrics.items():
                        self.logger.add_scalar(f'train/{k}', v, self.step)
                    self.logger.add_scalar(
                        'train/lr', self.scheduler.get_last_lr()[0], self.step
                    )

            # Valid
            self.model.eval()
            overall_metrics = defaultdict(list)
            for video in valid_videos:
                loader = DataLoader(
                    video,
                    shuffle=False,
                    batch_size=8,
                    num_workers=4,
                    collate_fn=lambda x: x,
                )
                name = video.info['name']
                avg_metrics = defaultdict(list)
                for graphs in tqdm(iter(loader), desc='Valid ' + name):
                    with torch.no_grad():
                        graphs = [graph.to('cuda') for graph in graphs]
                        output_dict, loss_dict = self.model(graphs)
                    metrics = self.valid_log_fn(graphs, output_dict, loss_dict)
                    for k, v in metrics.items():
                        avg_metrics[k].append(v)
                for k, v in avg_metrics.items():
                    avg = sum(v) / len(v)
                    self.logger.add_scalar(f'{name}/{k}', avg, self.epoch)
                    overall_metrics[k].append(avg)
            for k, v in overall_metrics.items():
                self.logger.add_scalar(f'OVERALL/{k}', sum(v) / len(v), self.epoch)

            # Track
            self.model.eval()
            names = []
            df_preds = []
            df_trues = []
            for video in valid_videos:
                name = video.info['name']
                df_pred = self.track_fn(self.model, video, use_gt=False)
                df_pred.to_csv(epoch_dir / f'{name}.txt', index=None, header=None)
                df_true = video.load_gt()
                names.append(name)
                df_preds.append(df_pred)
                df_trues.append(df_true)
            summary = mot_metrics(df_preds, df_trues, names)
            for name, metrics in summary.items():
                for k, v in metrics.items():
                    self.logger.add_scalar(f'{name}/{k}', v, self.epoch)

            # Other
            self.scheduler.step()
            torch.save(self.model, epoch_dir / 'model.pth')

        torch.save(self.model, epoch_dir / 'model.pth')

