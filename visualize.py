import os
import sys
import gzip
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

sys.path.append(os.path.join(os.getcwd(), 'modules'))
sys.path.append(os.path.join(os.getcwd(), 'modules/hplots'))
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':' + os.path.join(os.getcwd(), 'modules')
from OCHits2Showers import OCHits2ShowersLayer, process_endcap, OCGatherEnergyCorrFac
from ShowersMatcher import ShowersMatcher
from hgcal_analysis_plotter import HGCalAnalysisPlotter

prediction_path = '/Data/pred_202_nanoML.bin.gz'
with gzip.open(prediction_path) as f:
    prediction = pickle.load(f)
features_dict, truth_dict, predictions_dict = prediction[0]

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
dataframe = showers_matcher.get_result_as_dataframe()