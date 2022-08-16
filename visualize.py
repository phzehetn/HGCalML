import pdb

import os
import sys
import gzip
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.join(os.getcwd(), 'modules'))
sys.path.append(os.path.join(os.getcwd(), 'modules/hplots'))
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':' + os.path.join(os.getcwd(), 'modules')
from OCHits2Showers import OCHits2ShowersLayer, process_endcap, OCGatherEnergyCorrFac
from ShowersMatcher import ShowersMatcher
from hgcal_analysis_plotter import HGCalAnalysisPlotter

def add_dict_to_df(df, dictionary):
    print(dictionary.keys())
    for key in dictionary.keys():
        value = dictionary[key]
        print(key)
        print(value.shape)
        if len(value.shape) != 2:
            print("Unexpected dimension for key", key)
            continue
        # if value.shape[1] == 0:
        #     print("Nothing stored in key", key)
        if value.shape[1] == 1:
            df[key] = value.flatten()
        elif value.shape[1] > 1:
            for j in range(value.shape[1]):
                df[key+'_'+str(j)] = value[:,j].flatten()
        else:
            raise ValueError
    return df


def pred_to_full_dfs(predictions, n=-1):
    
    if n != -1:
        predictions = predictions[:n]

    df = pd.DataFrame()
    for i, (features, truth, prediction) in enumerate(predictions):
        tmp_df = pd.DataFrame()

        tmp_df = add_dict_to_df(tmp_df, features)
        tmp_df = add_dict_to_df(tmp_df, truth)
        tmp_df = add_dict_to_df(tmp_df, prediction)
        tmp_df['event_index'] = i

        df = pd.concat([df, tmp_df])

    return df


prediction_path = '/Data/pred_202_nanoML.bin.gz'
with gzip.open(prediction_path) as f:
    prediction = pickle.load(f)
features_dict, truth_dict, predictions_dict = prediction[0]

full_df = pred_to_full_dfs(prediction, n=1)

beta_threshold = 0.1
distance_threshold = 0.5
iou_threshold = 0.1
matching_mode = 'iou_max'
local_distance_scaling = True
de_e_cut = -1
angle_cut = -1

hits2showers = OCHits2ShowersLayer(beta_threshold, distance_threshold, local_distance_scaling)
showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut)
energy_gatherer = OCGatherEnergyCorrFac()

processed_pred_dict, pred_shower_alpha_idx = process_endcap(
    hits2showers, energy_gatherer, features_dict, predictions_dict)
showers_matcher.set_inputs(
    features_dict=features_dict,
    truth_dict=truth_dict,
    predictions_dict=processed_pred_dict,
    pred_alpha_idx=pred_shower_alpha_idx
)
showers_matcher.process()
showers_df = showers_matcher.get_result_as_dataframe()

full_df = pd.DataFrame()
for d in [features_dict, truth_dict, processed_pred_dict]:
    full_df = add_dict_to_df(full_df, d)


def matcher(full_df):
    return full_df


'''
def draw_sliders(full_df):
    fig = go.Figure()

    truth_IDs = full_df.truthHitAssignementIdx.unique()
    pred_IDs = full_df.pred_sid.unique()
    n_truth = truth_IDs.shape[0]
    n_pred = pred_IDs.shape[0]

    for id in truth_IDs:
        tmp_df = full_df[full_df.truthHitAssignementIDx == id]
        fig.add_trace(
            go.Scatter3D(
                visible=False,
                name="Shower " + str(id),
                x=tmp_df.RecHitX,
                y=tmp_df.RecHitY,
                z=tmp_df.RecHitZ,
            )
        )
    return
'''


fig = px.scatter_3d(full_df, 
    x='recHitX', y='recHitY', z='recHitZ', 
    size=0.2 + np.log(full_df['recHitEnergy'].to_numpy()+1), 
    animation_group='truthHitAssignementIdx', 
    range_x=[-145.0, 220], range_y=[-145.0, 110], range_z=[-520.0, -330.0], 
    color='recHitEnergy', alpha=0.5,
)
fig.add_trace(
    go.Scatter3d(
        visible=True,
        x=full_df.recHitX,
        y=full_df.recHitY,
        z=full_df.recHitZ,
        mode='markers',
        marker={
            'color': full_df.truthHitAssignementIdx,
            'size': 0.3,
            'symbol': 'diamond',
        }
    )
)
fig.write_html('../../Data/test.html')

print("DONE")

