import numpy as np
import pandas as pd
import os
import argparse
import torch
from tqdm import tqdm

from collections import defaultdict, OrderedDict
from natsort import natsorted
from pathlib import Path

def read_expr(expr_path):
    metrics = defaultdict(list)
    score_paths = os.listdir(expr_path)
    score_paths = [x for x in score_paths if '_scores.pth' in x]
    score_paths = natsorted(score_paths)

    for score_path in score_paths:
        scores = torch.load(expr_path / score_path, map_location=torch.device('cpu'))
        metrics['score_min'].append(scores.diag().min().item())
        frow = torch.nn.Softmax(dim=0)
        fcol = torch.nn.Softmax(dim=1)
        metrics['score_con'].append(
            np.asarray(frow(scores.float() / 0.07).diag().mean().item() + fcol(scores.float() / 0.07).diag().mean().item()) / 2.0
        )

    return metrics 


results_dir = Path('results')
save_path = 'qual.csv'
df = pd.DataFrame()

if __name__ == "__main__":
    dirs = ["C:\\Users\\ferdawsi\\Documents\\EECS_542\\va_reproduction\\"]
    res = defaultdict(list)
    conceals, aligns = [], []
    for dir in tqdm(dirs):
        dir_path = Path(dir)
        score_mins, score_cons = [], []
        expr_names = [x for x in os.listdir(dir_path) if (dir_path/ x).is_dir()]
        expr_names = natsorted(expr_names)

        for i, expr_name in enumerate(expr_names):
            expr_path = dir_path / expr_name
            metrics = read_expr(expr_path)
            score_mins += metrics['score_min']
            score_cons += metrics['score_con']
            aligns.append(np.array(metrics['score_min']).mean())
            conceals.append(np.array(metrics['score_con']).mean())
        res['avg_min'].append(np.array(score_mins).mean())
        res['avg_min90'].append(np.percentile(np.array(score_mins), 90))
        res['avg_min95'].append(np.percentile(np.array(score_mins), 95))
        res['avg_con'].append(np.array(score_cons).mean())
        res['avg_con90'].append(np.percentile(np.array(score_cons), 90))
        res['avg_con95'].append(np.percentile(np.array(score_cons), 95))

    df = pd.DataFrame(OrderedDict(res.items()))
    # df.index = ['burgert', 'tancik', 'ours']
    df.index = ['va_rep']
    df.to_csv(save_path, sep='\t')
