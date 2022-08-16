import pdb
import os
import sys
import gzip
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


N = 0
if len(sys.argv) > 1:
    prediction_file = sys.argv[1]
else:
    prediction_file = "/home/philipp/Code/HGCalML/data/pred_110_nanoML.bin.gz"
    prediction_file = "/home/philipp/Code/HGCalML/data/store/pred_1827_nanoML.bin.gz"


with gzip.open(prediction_file, "rb") as f:
    predictions = pickle.load(f)

features, truth, prediction = predictions[N]
feature_columns = [
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
truth_columns = [
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
    ]
prediction_columns = [
    'pred_beta', 
    'pred_ccoords', 
    'pred_energy_corr_factor', 
    'pred_pos', 
    'pred_time', 
    'pred_id', 
    'pred_dist', 
    'row_splits',
]
df_features = pd.DataFrame()
df_truth = pd.DataFrame()
df_pred = pd.DataFrame()
for k in feature_columns:
    df_features[k] = np.squeeze(features[k])
for k in truth_columns:
    df_truth[k] = np.squeeze(truth[k])
for k in prediction_columns:
    df_pred[k] = np.squeeze(prediction[k])



def plot_rechits(features):
    fig = px.scatter_3d(
        data_frame=df_features,
        x='recHitX', 
        y='recHitY', 
        z='recHitZ', 
        color='recHitEnergy',
    ) 
    return fig



if __name__ == '__main__':

    fig = plot_rechits(features=features)
    fig.show()