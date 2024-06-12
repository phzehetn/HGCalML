import pdb
import os
import sys
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASEPATH = '/home/philipp/FI2'
BASEPATH = '/mnt/ceph/users/pzehetner/Paper/predictions'
DATE = '0918'
ANALYSISFILE = 'analysis_test_0.5_fullscan.bin.gz'
ANALYSISFILE = 'analysis_hdbscan.bin.gz'
ANALYSISFILE = 'classic_d025_i05.bin.gz'
OUTFILE = os.path.join(BASEPATH, 'pu_results_200.pkl')

DIRS = [d for d in os.listdir(BASEPATH) if d.startswith(DATE)]
DIRS = [d for d in DIRS if os.path.exists(os.path.join(BASEPATH, d, ANALYSISFILE))]

summary = {}
dataframes = []

for i, d in enumerate(DIRS):

    print(f"Scanning directory {d}")
    FILE = os.path.join(BASEPATH, d, ANALYSISFILE)
    e_true, e_pred = [], []
    matched, unmatched = [], []
    with gzip.open(FILE, 'rb') as f:
        showers = pickle.load(f)['showers_dataframe']
    n_events = np.max(showers['event_id']) + 1
    dataframe = pd.DataFrame()
    for event in range(n_events):
        shower = showers[(showers['event_id'] == event) & (showers['truthHitAssignementIdx'] == 0)]
        dataframe = pd.concat((dataframe, shower))
        if shower.shape[0] != 1:
            # pdb.set_trace()
            print(d, "   ", event)

        if not shower.pred_energy_hits.isna().any():
            e_true.append(shower.truthHitAssignedEnergies.iloc[0])
            e_pred.append(shower.pred_energy_hits.iloc[0])
            matched.append(event)
        else:
            unmatched.append(event)

    summary[d] = {
        'e_true': np.array(e_true),
        'e_pred': np.array(e_pred),
        'matched': np.array(matched),
        'unmatched': np.array(unmatched),
        'showers': dataframe,
    }

with open(OUTFILE, 'wb') as f:
    pickle.dump(summary, f)

