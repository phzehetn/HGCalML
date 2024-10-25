import pdb
import pickle
import gzip
import os
import argparse
import numpy as np
import pandas as pd
from visualize_event import djcdc_to_dataframe 

import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Compare original data with preclustered data')
    parser.add_argument('--full', help='DataCollection for original data set')
    parser.add_argument('--pre', help='DataCollection for preclustered dat set')
    parser.add_argument('--out', help='Output Directory')
    parser.add_argument('--nevents', help='Maximum number of events, not yet used', default=1)
    args = parser.parse_args()

    INPUT_FULL = args.full
    INPUT_PRE = args.pre
    OUTPUT = args.out
    HGCALML = os.environ['HGCALML']

    if HGCALML is None and (INPUT_FULL.endswith('.djcdc') or INPUT_PRE.endswith('.djcdc')):
        print("HGCALML not set, cannot work with .djcdc files")
        sys.exit(1)

    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    if INPUT_FULL.endswith('.djcdc'):
        df_full = djcdc_to_dataframe(INPUT_FULL, 10)
    elif INPUT_FULL.endswith('.bin.gz'):
        with gzip.open(INPUT_FULL, 'rb') as f:
            truth, features = pickle.load(f)
        features = features.drop(columns=['event_id'])
        df_full = pd.concat([features, truth], axis=1)
    else:
        print("Unknown file type")
        sys.exit(1)

    if INPUT_PRE.endswith('.djcdc'):
        df_pre = djcdc_to_dataframe(INPUT_PRE, 10)
    elif INPUT_PRE.endswith('.bin.gz'):
        with gzip.open(INPUT_PRE, 'rb') as f:
            truth, features = pickle.load(f)
        features = features.drop(columns=['event_id'])
        df_pre = pd.concat([features, truth], axis=1)
    else:
        print("Unknown file type")
        sys.exit(1)

    columns = df_full.columns

    n_removed_showers = []
    e_removed_showers = []


    pre_pid = []
    pre_deposited = []
    pre_t_deposited = []
    pre_truth_energy = []
    full_pid = []
    full_deposited = []
    full_t_deposited = []
    full_truth_energy = []

    showers_lost = []

    for event in range(min(int(args.nevents), df_full.event_id.max(), df_pre.event_id.max())):
        print(event)
        particle_ids = np.unique(df_full.truthHitAssignementIdx)
        df_full_i = df_full[df_full.event_id == event]
        df_pre_i = df_pre[df_pre.event_id == event]
        for pid in particle_ids:
            if pid == -1: continue
            df_full_pid = df_full_i[df_full_i.truthHitAssignementIdx == pid]
            df_pre_pid = df_pre_i[df_pre_i.truthHitAssignementIdx == pid]
            if df_full_pid.shape[0] == 0: continue
            if df_pre_pid.shape[0] == 0: 
                showers_lost.append(df_full_pid['truthHitAssignedEnergies'].iloc[0])
                continue
            if not np.all(df_full_pid.truthHitAssignedPIDs == df_full_pid.truthHitAssignedPIDs.iloc[0]):
                pdb.set_trace()
            assert np.all(df_full_pid.truthHitAssignedPIDs == df_full_pid.truthHitAssignedPIDs.iloc[0]), "mismatch in full df"
            assert np.all(df_pre_pid.truthHitAssignedPIDs == df_pre_pid.truthHitAssignedPIDs.iloc[0]), "mismatch in full df"
            
            pre_pid.append(df_pre_pid.truthHitAssignedPIDs.iloc[0])
            full_pid.append(df_full_pid.truthHitAssignedPIDs.iloc[0])
            pre_deposited.append(df_pre_pid[df_pre_pid.recHitID == 0].recHitEnergy.sum())
            full_deposited.append(df_full_pid[df_full_pid.recHitID == 0].recHitEnergy.sum())
            pre_t_deposited.append(df_pre_pid.t_rec_energy.iloc[0])
            full_t_deposited.append(df_full_pid.t_rec_energy.iloc[0])
            pre_truth_energy.append(df_pre_pid.truthHitAssignedEnergies.iloc[0])
            full_truth_energy.append(df_full_pid.truthHitAssignedEnergies.iloc[0])


    data_dict = {
        'pre_pid': pre_pid,
        'full_pid': full_pid,
        'pre_deposited': pre_deposited,
        'full_deposited': full_deposited,
        'pre_t_deposited': pre_t_deposited,
        'full_t_deposited': full_t_deposited,
        'pre_truth_energy': pre_truth_energy,
        'full_truth_energy': full_truth_energy,
        'showers_lost': showers_lost,
        }
    with gzip.open(os.path.join(args.out, 'data.bin.gz'), "wb") as f:
        pickle.dump(data_dict, f)
    df = pd.DataFrame({
        'pre_pid': pre_pid,
        'full_pid': full_pid,
        'pre_deposited': pre_deposited,
        'full_deposited': full_deposited,
        'pre_t_deposited': pre_t_deposited,
        'full_t_deposited': full_t_deposited,
        'pre_truth_energy': pre_truth_energy,
        'full_truth_energy': full_truth_energy,
        })

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
    ax[0].scatter(df.full_t_deposited, df.full_deposited)
    ax[0].set_xlabel("t_rec_energy", fontsize=18)
    ax[0].set_ylabel("recHitEnergy[~track].sum()", fontsize=18)
    ax[0].set_title("Original Data Set", fontsize=20)
    ax[0].set_xlim(0, 250)
    ax[0].set_ylim(0, 250)

    ax[1].scatter(df.pre_t_deposited, df.pre_deposited)
    ax[1].set_xlabel("t_rec_energy", fontsize=18)
    ax[1].set_ylabel("recHitEnergy[~track].sum()", fontsize=18)
    ax[1].set_title("Preclustered Data Set", fontsize=20)
    ax[1].set_xlim(0, 250)
    ax[1].set_ylim(0, 250)
    plt.savefig(os.path.join(args.out, "deposited.jpg"))


    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,20))
    ax = ax.flatten()

    ax[0].scatter(df.full_pid, df.pre_pid)
    ax[0].set_title("Check if PID matches", fontsize=20)
    ax[0].set_xlabel("Original PID", fontsize=20)
    ax[0].set_ylabel("Pre-clustered PID", fontsize=20)

    ax[1].scatter(df.full_deposited, df.pre_deposited)
    ax[1].set_title("Deposited Energy (sum of `recHitEnergy`)", fontsize=20)
    ax[1].set_xlabel("Original", fontsize=20)
    ax[1].set_ylabel("Pre-clustered", fontsize=20)

    ax[2].scatter(df.full_t_deposited, df.pre_t_deposited)
    ax[2].set_title("Deposited Energy (`t_rec_energy`)", fontsize=20)
    ax[2].set_xlabel("Original", fontsize=20)
    ax[2].set_ylabel("Pre-clustered", fontsize=20)

    ax[3].scatter(df.full_truth_energy, df.pre_truth_energy)
    ax[3].set_title("True Energy (`truthHitAssignedEnergies`)", fontsize=20)
    ax[3].set_xlabel("Original", fontsize=20)
    ax[3].set_ylabel("Pre-clustered", fontsize=20)

    plt.savefig(os.path.join(args.out, "scatter.jpg"))




