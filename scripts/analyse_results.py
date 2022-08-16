import pdb

import os
from socket import TIPC_MEDIUM_IMPORTANCE
import sys
import gzip
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

f_dict_keys = [
    'recHitEnergy', 
    'recHitEta', 
    'recHitID', 
    'recHitTheta', 
    'recHitR', 
    'recHitX', 
    'recHitY', 
    'recHitZ', 
    'recHitTime', 
    'recHitHitR', 
    ]

t_dict_keys = [
    'truthHitAssignementIdx', 
    'truthHitAssignedEnergies', 
    'truthHitAssignedX', 
    'truthHitAssignedY', 
    'truthHitAssignedZ', 
    'truthHitAssignedEta', 
    'truthHitAssignedPhi', 
    'truthHitAssignedT', 
    'truthHitAssignedPIDs', 
    'truthHitSpectatorFlag', 
    'truthHitFullyContainedFlag', 
    't_rec_energy'
    ]

p_dict_keys = [
    'pred_beta',
    'pred_ccoords', 
    'pred_energy_corr_factor', 
    'pred_energy_low_quantile', 
    'pred_energy_high_quantile', 
    'pred_pos', 
    'pred_time', 
    'pred_id', 
    'pred_dist', 
    # 'row_splits'
    ]


def pred_to_full_dfs(predictions, n=-1):
    if n != -1:
        predictions = predictions[:n]
    list_feat = []
    list_truth = []
    list_pred = []

    for i, (features, truth, prediction) in enumerate(predictions):
        tmp_feat = pd.DataFrame()
        tmp_truth = pd.DataFrame()
        tmp_pred = pd.DataFrame()

        for col in f_dict_keys:
            tmp_feat[col] = np.squeeze(features[col])
        for col in t_dict_keys:
            tmp_truth[col] = np.squeeze(truth[col])
        for col in p_dict_keys:
            vals = np.squeeze(prediction[col])
            if len(vals.shape) > 1:
                for j in range(vals.shape[1]):
                    tmp_pred[f"{col}_{j}"] = vals[:,j]
            else:
                tmp_pred[col] = vals

        list_feat.append(tmp_feat)
        list_truth.append(tmp_truth)
        list_pred.append(tmp_pred)
    
    df_feat = pd.concat(list_feat)
    df_truth = pd.concat(list_truth)
    df_pred = pd.concat(list_pred)

    return df_feat, df_truth, df_pred


def overview_f_dict(f_dict, path=None):
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20,10))
    ax = ax.flatten()

    for i, key in enumerate(f_dict.keys()):
        ax[i].hist(f_dict[key], bins=50, 
            label=f"Mean: {np.mean(f_dict[key]):.2f}")
        ax[i].set_title(key, fontsize=20)
        ax[i].legend()
    fig.suptitle('features', fontsize=30)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig


def overview_t_dict(t_dict, path=None):
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20,10))
    ax = ax.flatten()

    for i, key in enumerate(t_dict.keys()):
        ax[i].hist(t_dict[key], bins=50, 
            label=f"Mean: {np.mean(t_dict[key]):.2f}")
        ax[i].set_title(key, fontsize=20)
        ax[i].legend()
    fig.suptitle('truth', fontsize=30)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig


def overview_p_dict(df_pred, path=None):
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(20,10))
    ax = ax.flatten()

    for i, key in enumerate(df_pred.columns):
        ax[i].hist(df_pred[key], bins=50, 
            label=f"Mean: {np.mean(df_pred[key]):.2f}")
        ax[i].set_title(key, fontsize=20)
        ax[i].legend()
    fig.suptitle('predictions', fontsize=30)
    fig.tight_layout()
    if path is not None:
        print(f"saving to {path}")
        fig.savefig(path)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analysis')
    parser.add_argument("prediction", 
        help='Predictions file (ends with .bin.gz)',
        )
    parser.add_argument('output_dir',
        help='Directory where plots will be stored',
        )
    args = parser.parse_args()

    with gzip.open(args.prediction, 'rb') as f:
        prediction = pickle.load(f)

    df_feat, df_truth, df_pred = pred_to_full_dfs(prediction, n=10)

    overview_f_dict(df_feat, os.path.join(args.output_dir, 'features.png'))
    overview_t_dict(df_truth, os.path.join(args.output_dir, 'truth.png'))
    overview_p_dict(df_pred, os.path.join(args.output_dir, 'predictions.png'))
