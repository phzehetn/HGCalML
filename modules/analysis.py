#TODO:  Load one sample of test data
#TODO:  Load pre-selection model and apply it to test data
#TODO:  Visualize data and pre-selection data
#TODO:  Load pre-selection model and apply it to preselection-data
#TODO:  Load normal model and apply it to data
#TODO:  Visualization

import pdb

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
mpl.style.use('Style.mplstyle')

from datastructures.TrainData_NanoML import TrainData_NanoML
from datastructures.TrainData_PreselectionNanoML import TrainData_PreselectionNanoML
from OCHits2Showers import OCHits2ShowersLayer, process_endcap, OCGatherEnergyCorrFac
from ShowersMatcher import ShowersMatcher
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
from DeepJetCore.modeltools import load_model


def add_dict_to_df(df, dictionary):
    for key in dictionary.keys():
        value = dictionary[key]
        if len(value.shape) != 2:
            print("Unexpected dimension for key", key)
            continue
        # if value.shape[1] == 0:
        #     print("Nothing stored in key", key)
        if value.shape[1] == 1:
            if isinstance(value, tf.Tensor):
                df[key] = value.numpy().flatten()
            elif type(value) == np.ndarray:
                df[key] = value.flatten()
            else:
                print("Unexpected type! ", value.__class__)
        elif value.shape[1] > 1:
            for j in range(value.shape[1]):
                if isinstance(value, tf.Tensor):
                    df[key+'_'+str(j)] = value[:,j].numpy().flatten()
                elif type(value) == np.ndarray:
                    df[key+'_'+str(j)] = value[:,j].flatten()
                else:
                    print("Unexpected type! ", value.__class__)
        else:
            raise ValueError
    return df


NFILES = 1
SCATTER_3D = False
ENERGY_DISTRIBUTION = True
RESOLUTION = True

beta_threshold = 0.1
distance_threshold = 0.5
iou_threshold = 0.1
matching_mode = 'iou_max'
local_distance_scaling = True
de_e_cut = -1
angle_cut = -1

path_original_data = '/Data/ML4Reco/test_newformat/dataCollection.djcdc'
path_preselection_data = '/Data/ML4Reco/test_newformat_preselection/dataCollection.djcdc'
path_preselection_model = '/Code/HGCalML/models/pre_selection_june22/KERAS_model.h5'
path_standard_model = '/Data/ML4Reco/standard_model/KERAS_check_model_block_5_epoch_50.h5'
path_preselection_based_model = '/Data/ML4Reco/preselection_model/KERAS_check_model_block_5_epoch_150.h5'

orig_dc = DataCollection(path_original_data)
pres_dc = DataCollection(path_preselection_data)
orig_input_files = [orig_dc.dataDir + filename for filename in orig_dc.samples]
pres_input_files = [pres_dc.dataDir + filename for filename in pres_dc.samples]
if len(orig_input_files) > NFILES:
    orig_input_files = orig_input_files[:NFILES]
if len(pres_input_files) > NFILES:
    pres_input_files = pres_input_files[:NFILES]

pre_model = load_model(path_preselection_model)
standard_model = load_model(path_standard_model)
preselection_model = load_model(path_preselection_based_model)
    
all_orig_rechits = np.array([])
all_pres_rechits = np.array([])
all_orig_truth = np.array([])
all_orig_pred = np.array([])
all_pres_truth = np.array([])
all_pres_pred = np.array([])

for n in range(NFILES):
    orig_td = orig_dc.dataclass()
    orig_td.readFromFileBuffered(orig_input_files[n])
    pres_td = pres_dc.dataclass()
    pres_td.readFromFileBuffered(pres_input_files[n])

    orig_gen = TrainDataGenerator()
    orig_gen.setBatchSize(1)
    orig_gen.setSquaredElementsLimit(False)
    orig_gen.setSkipTooLargeBatches(False)
    orig_gen.setBuffer(orig_td)
    orig_num_steps = orig_gen.getNBatches()
    orig_generator = orig_gen.feedNumpyData()

    pres_gen = TrainDataGenerator()
    pres_gen.setBatchSize(1)
    pres_gen.setSquaredElementsLimit(False)
    pres_gen.setSkipTooLargeBatches(False)
    pres_gen.setBuffer(pres_td)
    pres_num_steps = pres_gen.getNBatches()
    pres_generator = pres_gen.feedNumpyData()

    num_steps = min(orig_num_steps, pres_num_steps)
    print("STEPS: ", num_steps)


    for i in range(num_steps):
        print("Step ", i, " out of ", num_steps)
        orig_data_in = next(orig_generator)   # Tuple, second entry is empty list
        orig_predictions_dict = standard_model(orig_data_in[0])
        # Keys:     'pred_beta', 'pred_ccoords', 'pred_energy_corr_factor', 
        #           'pred_energy_low_quantile', 'pred_energy_high_quantile', 
        #           'pred_pos', 'pred_time', 'pred_id', 'pred_dist', 'row_splits'
        orig_features_dict = orig_td.createFeatureDict(orig_data_in[0])
        # Keys:     'recHitEnergy', 'recHitEta', 'recHitID', 'recHitTheta', 
        #           'recHitR', 'recHitX', 'recHitY', 'recHitZ', 'recHitTime', 
        #           'recHitHitR', 'recHitXY'
        orig_truth_dict = orig_td.createTruthDict(orig_data_in[0])
        # Keys:     'truthHitAssignementIdx', 'truthHitAssignedEnergies', 
        #           'truthHitAssignedX', 'truthHitAssignedY', 'truthHitAssignedZ', 
        #           'truthHitAssignedEta', 'truthHitAssignedPhi', 'truthHitAssignedT', 
        #           'truthHitAssignedPIDs', 'truthHitSpectatorFlag', 
        #           'truthHitFullyContainedFlag', 't_rec_energy'
        orig_df = pd.DataFrame()
        for d in [orig_predictions_dict, orig_features_dict, orig_truth_dict]:
            orig_df = add_dict_to_df(orig_df, d)

        orig_hits2showers = OCHits2ShowersLayer(beta_threshold, distance_threshold, local_distance_scaling)
        orig_showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut)
        orig_energy_gatherer = OCGatherEnergyCorrFac()
        orig_processed_pred_dict, orig_pred_shower_alpha_idx = process_endcap(
            orig_hits2showers, orig_energy_gatherer, orig_features_dict, orig_predictions_dict)
        orig_processed_pred_df = pd.DataFrame()
        orig_processed_pred_df = add_dict_to_df(orig_processed_pred_df, orig_processed_pred_dict)
        opp_df = orig_processed_pred_df
        orig_alpha_predictions = opp_df.iloc[orig_pred_shower_alpha_idx]
        orig_showers_matcher.set_inputs(
            features_dict=orig_features_dict,
            truth_dict=orig_truth_dict,
            predictions_dict=orig_processed_pred_dict,
            pred_alpha_idx=orig_pred_shower_alpha_idx
        )
        orig_showers_matcher.process()
        orig_showers_df = orig_showers_matcher.get_result_as_dataframe()
        orig_mask_truth_matched = orig_showers_df.truthHitAssignedX.notna().to_numpy()
        orig_mask_pred_matched = orig_showers_df.pred_energy_high_quantile.notna().to_numpy()
        orig_mask_fully_matched = np.logical_and(orig_mask_truth_matched, orig_mask_pred_matched)
        orig_truth_matched = orig_showers_df[orig_mask_truth_matched]
        orig_pred_matched = orig_showers_df[orig_mask_pred_matched]
        orig_fully_matched = orig_showers_df[orig_mask_fully_matched]
        
        # For plotting later
        all_orig_rechits = np.append(all_orig_rechits, orig_df.recHitEnergy.to_numpy())
        all_orig_truth = np.append(all_orig_truth, orig_fully_matched.truthHitAssignedEnergies)
        all_orig_pred = np.append(all_orig_pred, orig_fully_matched.pred_energy)
        
        pres_data_in = next(pres_generator)   # Tuple, second entry is empty list
        pres_predictions_dict = preselection_model(pres_data_in[0])    # Dictionary
        # Keys:     'pred_beta', 'pred_ccoords', 'pred_energy_corr_factor', 
        #           'pred_energy_low_quantile', 'pred_energy_high_quantile', 
        #           'pred_pos', 'pred_time', 'pred_id', 'pred_dist', 'row_splits'
        pres_features_dict = pres_td.createFeatureDict(pres_data_in[0])
        # Keys:     'recHitEnergy', 'recHitEta', 'recHitID', 'recHitTheta', 
        #           'recHitR', 'recHitX', 'recHitY', 'recHitZ', 'recHitTime', 
        #           'recHitHitR', 'recHitXY'
        pres_truth_dict = pres_td.createTruthDict(pres_data_in[0])
        # Keys:     'truthHitAssignementIdx', 'truthHitAssignedEnergies', 
        #           'truthHitAssignedX', 'truthHitAssignedY', 'truthHitAssignedZ', 
        #           'truthHitAssignedEta', 'truthHitAssignedPhi', 'truthHitAssignedT',  
        #           'truthHitAssignedPIDs', 'truthHitSpectatorFlag', 
        #           'truthHitFullyContainedFlag', 't_rec_energy'
        pres_df = pd.DataFrame()
        for d in [pres_predictions_dict, pres_features_dict, pres_truth_dict]:
            pres_df = add_dict_to_df(pres_df, d)

        pres_hits2showers = OCHits2ShowersLayer(beta_threshold, distance_threshold, local_distance_scaling)
        pres_showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut)
        pres_energy_gatherer = OCGatherEnergyCorrFac()
        pres_processed_pred_dict, pres_pred_shower_alpha_idx = process_endcap(
            pres_hits2showers, pres_energy_gatherer, pres_features_dict, pres_predictions_dict)
        pres_processed_pred_df = pd.DataFrame()
        pres_processed_pred_df = add_dict_to_df(pres_processed_pred_df, pres_processed_pred_dict)
        ppp_df = pres_processed_pred_df
        pres_showers_matcher.set_inputs(
            features_dict=pres_features_dict,
            truth_dict=pres_truth_dict,
            predictions_dict=pres_processed_pred_dict,
            pred_alpha_idx=pres_pred_shower_alpha_idx
        )
        pres_showers_matcher.process()
        pres_showers_df = pres_showers_matcher.get_result_as_dataframe()
        pres_mask_truth_matched = pres_showers_df.truthHitAssignedX.notna().to_numpy()
        pres_mask_pred_matched = pres_showers_df.pred_energy_high_quantile.notna().to_numpy()
        pres_mask_fully_matched = np.logical_and(pres_mask_truth_matched, pres_mask_pred_matched)
        pres_truth_matched = pres_showers_df[pres_mask_truth_matched]
        pres_pred_matched = pres_showers_df[pres_mask_pred_matched]
        pres_fully_matched = pres_showers_df[pres_mask_fully_matched]

        # For plotting
        all_pres_rechits = np.append(all_pres_rechits, pres_df.recHitEnergy.to_numpy())
        all_pres_truth = np.append(all_pres_truth, pres_fully_matched.truthHitAssignedEnergies)
        all_pres_pred = np.append(all_pres_pred, pres_fully_matched.pred_energy)

    all_orig_rechits = np.array(all_orig_rechits).flatten()
    all_pres_rechits = np.array(all_pres_rechits).flatten()
    all_orig_truth = np.array(all_orig_truth).flatten()
    all_orig_pred = np.array(all_orig_pred).flatten()
    all_pres_truth = np.array(all_pres_truth).flatten()
    all_pres_pred = np.array(all_pres_pred).flatten()

    # 3D Scatter plot
    if SCATTER_3D:
        hits_figure = go.Figure()
        hits_figure.add_trace(
            go.Scatter3d(
                name = 'Normal model: ' + str(orig_df.shape[0]) + ' entries',
                visible = True,
                showlegend = True,
                x = orig_df['recHitX'],
                y = orig_df['recHitY'],
                z = orig_df['recHitZ'],
                mode = 'markers',
                marker = dict(
                    # size = 0.5 + np.log(orig_df['recHitEnergy'].values.flatten()+1),
                    # size = 0.2 + orig_df['recHitEnergy'].values.flatten(),
                    # size = 1.0,
                    size = 1.0 + orig_df['recHitEnergy'].values.flatten(),
                    color = 'blue',
                    opacity = 0.5,
                ),
                hovertemplate = \
                    'recHitX=%{x}<br>recHitY=%{y}<br>recHitZ=%{z}<br>recHitEnergy=%{customdata[0]}<extra></extra>',
                customdata = orig_df['recHitEnergy'].to_numpy().reshape(-1, 1)
            )
        )
        hits_figure.add_trace(
            go.Scatter3d(
                name = 'Preselection: ' + str(pres_df.shape[0]) + ' entries',
                visible = True,
                showlegend = True,
                x = pres_df['recHitX'],
                y = pres_df['recHitY'],
                z = pres_df['recHitZ'],
                mode = 'markers',
                marker = dict(
                    # size = 0.5 + np.log(orig_df['recHitEnergy'].values.flatten()+1),
                    # size = 0.2 + orig_df['recHitEnergy'].values.flatten(),
                    # size = 3.0,
                    size = 2.0 + orig_df['recHitEnergy'].values.flatten(),
                    color = 'green',
                    opacity = 0.5,
                ),
                hovertemplate = \
                    'recHitX=%{x}<br>recHitY=%{y}<br>recHitZ=%{z}<br>recHitEnergy=%{customdata[0]}<extra></extra>',
                customdata = orig_df['recHitEnergy'].to_numpy().reshape(-1, 1)
            )
        )
        hits_figure.add_trace(
            go.Scatter3d(
                name = 'Standard truth matched showers: ' + str(orig_truth_matched.shape[0]) + ' entries',
                visible = True,
                showlegend = True,
                x = orig_truth_matched['truthHitAssignedX'],
                y = orig_truth_matched['truthHitAssignedY'],
                z = orig_truth_matched['truthHitAssignedZ'],
                mode = 'markers',
                marker = dict(
                    symbol = 'circle',
                    # size = 0.5 + np.log(orig_df['recHitEnergy'].values.flatten()+1),
                    # size = 0.2 + orig_df['recHitEnergy'].values.flatten(),
                    # size = 3.0,
                    size = orig_truth_matched['truthHitAssignedEnergies'].values.flatten() / 5.0,
                    color = 'orange',
                    opacity = 0.5,
                ),
                hovertemplate = \
                    'truthHitAssignedX=%{x}<br>truthHitAssignedY=%{y}<br>truthHitAssignedZ=%{z}<br>truthHitAssignedEnergies=%{customdata[0]}<extra></extra>',
                customdata = orig_truth_matched['truthHitAssignedEnergies'].to_numpy().reshape(-1, 1)
            )
        )
        hits_figure.add_trace(
            go.Scatter3d(
                name = 'Preselection truth matched showers: ' + str(pres_truth_matched.shape[0]) + ' entries',
                visible = True,
                showlegend = True,
                x = pres_truth_matched['truthHitAssignedX'],
                y = pres_truth_matched['truthHitAssignedY'],
                z = pres_truth_matched['truthHitAssignedZ'],
                mode = 'markers',
                marker = dict(
                    symbol = 'diamond',
                    # size = 0.5 + np.log(orig_df['recHitEnergy'].values.flatten()+1),
                    # size = 0.2 + orig_df['recHitEnergy'].values.flatten(),
                    # size = 3.0,
                    size = pres_truth_matched['truthHitAssignedEnergies'].values.flatten() / 5.0,
                    color = 'purple',
                    opacity = 0.5,
                ),
                hovertemplate = \
                    'truthHitAssignedX=%{x}<br>truthHitAssignedY=%{y}<br>truthHitAssignedZ=%{z}<br>truthHitAssignedEnergies=%{customdata[0]}<extra></extra>',
                customdata = pres_truth_matched['truthHitAssignedEnergies'].to_numpy().reshape(-1, 1)
            )
        )
        hits_figure.update_layout(
            paper_bgcolor = 'grey', 
            plot_bgcolor = 'grey',
        )
        hits_figure.update_layout()
        fd = hits_figure.to_dict()
        hits_figure.write_html('/Data/ML4Reco/Plots/hits_figure.html')

    # Energy 
    if ENERGY_DISTRIBUTION: # Only to enable folding
        fig, ax_1 = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        ax_2 = ax_1.twinx()
        n, bins, patches = ax_1.hist([all_orig_rechits, all_pres_rechits], bins=50)
        ax_1.cla()
        width = 0.4 * (bins[1] - bins[0])
        bins_shifted = bins + width
        ax_1.bar(bins[:-1], n[0], width, align='edge', color='darkslateblue', label='standard')
        ax_2.bar(bins_shifted[:-1], n[1], width, align='edge', color='forestgreen', label='preselection')
        ax_1.set_xlabel("Energy", fontsize=20)
        ax_1.set_ylabel("Without preselection", color='darkslateblue', fontsize=20)
        ax_1.tick_params(axis='y', labelcolor='darkslateblue')
        ax_2.set_ylabel("With preselection", color='forestgreen', fontsize=20, labelpad=25., rotation=270)
        ax_2.tick_params(axis='y', labelcolor='forestgreen')
        ax_1.set_yscale('log')
        ax_2.set_yscale('log')
        ax_2.set_ylim(ax_1.get_ylim())
        ax_1.grid(visible=True, color='darkslategrey', lw=2.)
        custom_lines = [
            mpl.lines.Line2D([0], [0], color='darkslateblue', lw=4),
            mpl.lines.Line2D([0], [0], color='forestgreen', lw=4)
        ]
        ax_1.legend(custom_lines, ['Standard', 'Preselection'], fontsize=20)
        ax_1.set_title("Energy Distribution ('recHitEnergy')", fontsize=30)
        # ax_1.grid(visible=True, color='skyblue', lw=2.)
        # ax_2.grid(visible=True, color='lightgreen', lw=2.)
        plt.tight_layout()
        plt.savefig('/Data/ML4Reco/Plots/Energies.png')
        plt.close(fig)


    if RESOLUTION: # Energy predictions
        LIMIT = 250
        orig_truth = all_orig_truth
        orig_pred = all_orig_pred
        orig_ratio = orig_pred / orig_truth
        orig_n = orig_ratio.shape[0]
        pres_truth = all_pres_truth
        pres_pred = all_pres_pred
        pres_ratio = pres_pred / pres_truth
        pres_n = pres_ratio.shape[0]
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
        ax[0].scatter(orig_truth, orig_pred, color='darkslateblue', 
            label='Original (' + str(orig_n) + ' entries)')
        ax[0].scatter(pres_truth, pres_pred, color='forestgreen', 
            label='Preselection (' + str(pres_n) + ' entries)')
        ax[0].set_xlabel("True Energy", fontsize=20)
        ax[0].set_ylabel("Predicted Energy", fontsize=20)
        ax[0].set_title("Truth vs Prediction (Scatter Plot)" , fontsize=30)
        # max_y = 1.05 * max(orig_truth.max(), orig_pred.max(), pres_truth.max(), pres_pred.max())
        ax[0].set_xlim([-5, LIMIT])
        ax[0].set_ylim([-5, LIMIT])
        ax[0].plot(np.linspace(0, LIMIT, 100), np.linspace(0, LIMIT, 100), 
            '--', lw=1.0, color='black', label='Ideal response')
        ax[0].legend(fontsize=15)
        ax1 = ax[1]
        ax2 = ax1.twinx()
        numbers, bins, patches = ax1.hist([orig_ratio, pres_ratio], density=True, bins=1000)
        ax1.cla()
        width = 0.4 * (bins[1] - bins[0])
        bins_shifted = bins + width
        ax1.bar(bins[:-1], numbers[0], width, align='edge', color='darkslateblue', label='Standard')
        ax2.bar(bins_shifted[:-1], numbers[1], width, align='edge', color='forestgreen', label='Preselection')
        ax1.grid(visible=True, color='skyblue', lw=2.)
        ax2.grid(visible=True, color='lightgreen', lw=2.)
        ax1.set_title("Response (Prediction / Truth)", fontsize=30)
        custom_lines = [
            mpl.lines.Line2D([0], [0], color='darkslateblue', lw=4),
            mpl.lines.Line2D([0], [0], color='forestgreen', lw=4)
        ]
        ax1.set_xlim([0,5])
        ax2.set_xlim([0,5])
        ax1.legend(custom_lines, ['Standard', 'Preselection'], fontsize=20)
        plt.tight_layout()
        plt.savefig('/Data/ML4Reco/Plots/Resolution.png')
        plt.close(fig)
