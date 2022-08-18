import pdb
import os
import pickle
import gzip
from tkinter import NS

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from DeepJetCore.modeltools import load_model
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
from OCHits2Showers import OCHits2ShowersLayer, process_endcap, OCGatherEnergyCorrFac
from ShowersMatcher import ShowersMatcher

configuration = {
    'beta_threshold' : 0.1,
    'distance_threshold' : 0.5,
    'iou_threshold' : 0.1,
    'matching_mode' : 'iou_max',
    'local_distance_scaling' : True,
    'de_e_cut' : -1,
    'angle_cut' : -1,
}

def add_dict_to_df(df, dictionary):
    for key in dictionary.keys():
        value = dictionary[key]
        if len(value.shape) != 2:
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
                    pass
        else:
            raise ValueError
    return df


class ComparisonPlotter():

    def __init__(self, 
        prediction1=None, prediction2=None, 
        model1=None, model2=None, 
        dc1=None, dc2=None, 
        Nfiles=-1, Nsteps=-1,
        name1="Model 1", name2="Model 2"):


        self.processed = False
        if (prediction1 is None and prediction2 is not None) or \
            (prediction1 is not None and prediction2 is None):
            print("Provide predictions for either both or none of the models")
            raise ValueError
        elif (prediction1 is not None and prediction2 is not None):
            # Initialize with path to prediction output
            self._check_and_load_inputs_predictions(prediction1, prediction2)
        else:
            # Initialize with models and data collection
            # This needs more time, because it applies the model on all events
            self._check_and_load_inputs_models(
                model1=model1, model2=model2, 
                dc1=dc1, dc2=dc2, 
                Nfiles=Nfiles, Nsteps=Nsteps)

        self.name1 = str(name1)
        self.name2 = str(name2)

    def _check_and_load_inputs_predictions(self, prediction1, prediction2):
        if (not os.path.exists(prediction1)) and (prediction1[:-7] == '.bin.gz'):
            print("Please provide valid path to compressed file '*.bin.gz'")
            raise ValueError
        if (not os.path.exists(prediction2)) and (prediction2[:-7] == '.bin.gz'):
            print("Please provide valid path to compressed file '*.bin.gz'")
            raise ValueError
        
        with gzip.open(prediction1, 'rb') as f:
            prediction1 = pickle.load(f)
        with gzip.open(prediction2, 'rb') as f:
            prediction2 = pickle.load(f)

        self.all_hits1 = pd.concat([pred[0] for pred in prediction1])
        self.all_hits2 = pd.concat([pred[0] for pred in prediction2])
        self.all_showers1 = pd.concat([pred[1] for pred in prediction1])
        self.all_showers2 = pd.concat([pred[1] for pred in prediction2])
        self.processed = True

        return

    def _check_and_load_inputs_models(self, 
        model1, model2, 
        dc1, dc2, 
        Nfiles, Nsteps):

        if not self._is_model_or_path(model1):
            print("model1 should be a model or a valid path to a model")
            raise ValueError
        if not self._is_model_or_path(model2):
            print("model2 should be a model or a valid path to a model")
            raise ValueError
        if not self._is_dc_or_path(dc1):
            print("dc1 should be a DataCollection or a valid path to one")
            raise ValueError
        if not (self._is_dc_or_path(dc2) or dc2 is None):
            print("dc2 should be None, a DataCollection or a valid path to one")
            raise ValueError

        if type(model1) is str:
            self.model1 = load_model(model1)
        else:
            self.model1 = model1
        if type(model2) is str:
            self.model2 = load_model(model2)
        else:
            self.model2 = model2
        if type(dc1) is str:
            self.dc1 = DataCollection(dc1)
        else:
            self.dc1 = dc1
        if dc2 is None:
            self.dc2 = dc1
        elif type(dc2) is str:
            self.dc2 = DataCollection(dc2)
        else:
            self.dc2 = dc2

        self.files_1 = [self.dc1.dataDir + f for f in self.dc1.samples]
        self.files_2 = [self.dc2.dataDir + f for f in self.dc2.samples]

        self._process(Nfiles=Nfiles, Nsteps=Nsteps)

        return

    def _is_model_or_path(self, model):
        model_class = \
            "<class 'tensorflow.python.keras.engine.functional.Functional'>"
        if type(model) is str:
            return os.path.exists(model)
        return str(model.__class__) == model_class

    def _is_dc_or_path(self, dc):
        dc_class = \
            "<class 'DeepJetCore.DataCollection.DataCollection'>"
        if type(dc) is str:
            return os.path.exists(dc)
        return str(dc.__class__) == dc_class


    def _process(self, Nfiles=1, Nsteps=-1):
        # Only relevant if initialized with two models and datacollection
        eventID = 0
        all_hits1 = pd.DataFrame()
        all_hits2 = pd.DataFrame()
        all_showers1 = pd.DataFrame()
        all_showers2 = pd.DataFrame()
        if Nfiles == -1:
            Nfiles = min(len(self.files_1), len(self.files_2))
        for n in range(Nfiles):
            print("Processing file ", str(n), " of ", str(Nfiles))
            td1 = self.dc1.dataclass()
            td2 = self.dc2.dataclass()
            td1.readFromFileBuffered(self.files_1[n])
            td2.readFromFileBuffered(self.files_2[n])

            gen1 = TrainDataGenerator()
            gen2 = TrainDataGenerator()
            gen1.setBatchSize(1)
            gen2.setBatchSize(1)
            gen1.setSquaredElementsLimit(False)
            gen2.setSquaredElementsLimit(False)
            gen1.setSkipTooLargeBatches(False)
            gen2.setSkipTooLargeBatches(False)
            gen1.setBuffer(td1)
            gen2.setBuffer(td2)

            num_steps1 = gen1.getNBatches()
            num_steps2 = gen2.getNBatches()
            generator1 = gen1.feedNumpyData()
            generator2 = gen2.feedNumpyData()
            if Nsteps == -1:
                Nsteps = min(num_steps1, num_steps2)

            for i in range(Nsteps):
                print("Processing step ", str(i), " of ", str(Nsteps))
                eventID += 1
                data1 = next(generator1)
                data2 = next(generator2)
                pred_dict1 = self.model1(data1[0])
                pred_dict2 = self.model2(data2[0])
                feat_dict1 = td1.createFeatureDict(data1[0])
                feat_dict2 = td2.createFeatureDict(data2[0])
                truth_dict1 = td1.createTruthDict(data1[0])
                truth_dict2 = td2.createTruthDict(data2[0])
                hits2showers1 = OCHits2ShowersLayer(
                    configuration['beta_threshold'], 
                    configuration['distance_threshold'], 
                    configuration['local_distance_scaling'])
                hits2showers2 = OCHits2ShowersLayer(
                    configuration['beta_threshold'], 
                    configuration['distance_threshold'], 
                    configuration['local_distance_scaling'])
                showers_matcher1 = ShowersMatcher(
                    configuration['matching_mode'],
                    configuration['iou_threshold'], 
                    configuration['de_e_cut'], 
                    configuration['angle_cut'])
                showers_matcher2 = ShowersMatcher(
                    configuration['matching_mode'],
                    configuration['iou_threshold'], 
                    configuration['de_e_cut'], 
                    configuration['angle_cut'])
                energy_gatherer1 = OCGatherEnergyCorrFac()
                energy_gatherer2 = OCGatherEnergyCorrFac()
                processed_pred_dict1, pred_shower_alpha_idx1 = process_endcap(
                    hits2showers1, energy_gatherer1, feat_dict1, pred_dict1)
                processed_pred_dict2, pred_shower_alpha_idx2 = process_endcap(
                    hits2showers2, energy_gatherer2, feat_dict2, pred_dict2)
                showers_matcher1.set_inputs(
                    features_dict=feat_dict1,
                    truth_dict=truth_dict1,
                    predictions_dict=processed_pred_dict1,
                    pred_alpha_idx=pred_shower_alpha_idx1
                )
                showers_matcher2.set_inputs(
                    features_dict=feat_dict2,
                    truth_dict=truth_dict2,
                    predictions_dict=processed_pred_dict2,
                    pred_alpha_idx=pred_shower_alpha_idx2
                )
                showers_matcher1.process()
                showers_matcher2.process()

                showers_df1 = showers_matcher1.get_result_as_dataframe()
                showers_df2 = showers_matcher2.get_result_as_dataframe()
                showers_df1['eventID'] = eventID
                showers_df2['eventID'] = eventID
                hits_df1 = pd.DataFrame()
                hits_df2 = pd.DataFrame()
                for d in [processed_pred_dict1, feat_dict1, truth_dict1]:
                    hits_df1 = add_dict_to_df(hits_df1, d)
                for d in [processed_pred_dict2, feat_dict2, truth_dict2]:
                    hits_df2 = add_dict_to_df(hits_df2, d)
                hits_df1['eventID'] = eventID
                hits_df2['eventID'] = eventID

                all_hits1 = pd.concat([all_hits1, hits_df1])
                all_hits2 = pd.concat([all_hits2, hits_df2])
                all_showers1 = pd.concat([all_showers1, showers_df1])
                all_showers2 = pd.concat([all_showers2, showers_df2])

        self.all_showers1 = all_showers1
        self.all_showers2 = all_showers2
        self.all_hits1 = all_hits1
        self.all_hits2 = all_hits2
        self.processed = True
        return

###############################################################################
### Plotting functions start here #############################################
###############################################################################


    def energy_distribution(self, nbins=50, title1='', title2=''):
        if not self.processed:
            self._process()
        rechits1 = self.all_hits1['recHitEnergy'].to_numpy()
        rechits2 = self.all_hits2['recHitEnergy'].to_numpy()
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        ax2 = ax1.twinx()
        n, bins, patches = ax1.hist([rechits1, rechits2], bins=nbins)
        ax1.cla()
        width = 0.4 * (bins[1] - bins[0])
        bins_shifted = bins + width
        ax1.bar(bins[:-1], n[0], width, align='edge', 
            color='darkslateblue', label='standard')
        ax2.bar(bins_shifted[:-1], n[1], width, align='edge', 
            color='forestgreen', label='preselection')
        ax1.set_xlabel("Energy", fontsize=20)
        ax1.set_ylabel("Without preselection", color='darkslateblue', fontsize=20)
        ax1.tick_params(axis='y', labelcolor='darkslateblue')
        ax2.set_ylabel("With preselection", 
            color='forestgreen', fontsize=20, labelpad=25., rotation=270)
        ax2.tick_params(axis='y', labelcolor='forestgreen')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax2.set_ylim(ax1.get_ylim())
        ax1.grid(visible=True, color='darkslategrey', lw=2.)
        custom_lines = [
            mpl.lines.Line2D([0], [0], color='darkslateblue', lw=4),
            mpl.lines.Line2D([0], [0], color='forestgreen', lw=4) ]
        ax1.legend(custom_lines, [title1, title2], fontsize=20)
        ax1.set_title("Energy Distribution ('recHitEnergy')", fontsize=30)
        return fig, (ax1, ax2)


    def _resolution_plot(self, energy_range=None, figax=None):
        if not self.processed:
            self._process()
        if figax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        else:
            fig, ax = figax
        mask_truth1 = self.all_showers1.truthHitAssignedX.notna().to_numpy()
        mask_pred1 = self.all_showers1.pred_energy_high_quantile.notna().to_numpy()
        mask1 = np.logical_and(mask_truth1, mask_pred1)
        mask_truth2 = self.all_showers2.truthHitAssignedX.notna().to_numpy()
        mask_pred2 = self.all_showers2.pred_energy_high_quantile.notna().to_numpy()
        mask2 = np.logical_and(mask_truth2, mask_pred2)
        showers1 = self.all_showers1[mask1]
        showers2 = self.all_showers2[mask2]
        if energy_range is not None:
            showers1 = showers1[
                (showers1.truthHitAssignedEnergies > energy_range[0]) &
                (showers1.truthHitAssignedEnergies < energy_range[1]) ]
            showers2 = showers2[
                (showers2.truthHitAssignedEnergies > energy_range[0]) &
                (showers2.truthHitAssignedEnergies < energy_range[1]) ]
        else:
            energy_range = ("0", "Inf")
        truth1 = showers1.truthHitAssignedEnergies.to_numpy()
        pred1 = showers1.pred_energy.to_numpy()
        response1 = truth1 / pred1
        response1 = pred1 / truth1
        truth2 = showers2.truthHitAssignedEnergies.to_numpy()
        pred2 = showers2.pred_energy.to_numpy()
        response2 = pred2 / truth2
        axt = ax.twinx()
        numbers, bins, patches = ax.hist([response1, response2], density=True, bins=100)
        ax.cla()
        width = 0.4 * (bins[1] - bins[0])
        bins_shifted = bins + width
        label1 = self.name1 + " Energy: " + str(energy_range[0]) \
            + " GeV - " + str(energy_range[1]) + " GeV"
        label2 = self.name2 + " Energy: " + str(energy_range[0]) \
            + " GeV - " + str(energy_range[1]) + " GeV"
        ax.bar(bins[:-1], numbers[0], width, 
            align='edge', color='darkslateblue', label=label1,)
        axt.bar(bins_shifted[:-1], numbers[1], width, 
            align='edge', color='forestgreen', label=label2,)
        custom_lines = [
            mpl.lines.Line2D([0], [0], color='darkslateblue', lw=4),
            mpl.lines.Line2D([0], [0], color='forestgreen', lw=4)
        ]
        ax.legend(custom_lines, [label1, label2], fontsize=20)
        return fig, (ax, axt)

    def resolution_plots(self):
        ranges = [(0, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500)]
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(40,20))
        ax = ax.flatten()
        N = min(ax.shape[0], len(ranges))
        for i in range(N):
            fig, axes = self._resolution_plot(energy_range=ranges[i], figax=(fig, ax[i]))
        return fig, axes


    def matched_showers(self):
        N_events = self.all_showers1.eventID.max()
        ratios1 = []
        ratios2 = []
        for i in range(1, N_events+1):
            matched1 = self.all_showers1[self.all_showers1.eventID == i]
            matched2 = self.all_showers2[self.all_showers2.eventID == i]
            matched1 = matched1[matched1.truthHitAssignedX.notna()]
            matched2 = matched2[matched2.truthHitAssignedX.notna()]
            pred_matched1 = matched1[matched1.pred_pos.notna()]
            pred_matched2 = matched2[matched2.pred_pos.notna()]
            true_e1 = matched1.t_rec_energy.to_numpy()
            matched_e1 = pred_matched1.t_rec_energy.to_numpy()
            true_e2 = matched2.t_rec_energy.to_numpy()
            matched_e2 = pred_matched2.t_rec_energy.to_numpy()
            ratios1.append(matched_e1.sum() / true_e1.sum())
            ratios2.append(matched_e2.sum() / true_e2.sum())
        
        ratios1 = np.array(ratios1)
        ratios2 = np.array(ratios2)
        assert np.logical_and(np.all(ratios1 >= 0), np.all(ratios1 <= 1))
        assert np.logical_and(np.all(ratios2 >= 0), np.all(ratios2 <= 1))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        axt = ax.twinx()

        numbers, bins, patches = ax.hist([ratios1, ratios2], density=True, bins=30)
        ax.cla()
        width = 0.4 * (bins[1] - bins[0])
        bins_shifted = bins + width
        ax.bar(bins[:-1], numbers[0], width, 
            align='edge', color='darkslateblue', label=self.name1,)
        axt.bar(bins_shifted[:-1], numbers[1], width, 
            align='edge', color='forestgreen', label=self.name2,)
        custom_lines = [
            mpl.lines.Line2D([0], [0], color='darkslateblue', lw=4),
            mpl.lines.Line2D([0], [0], color='forestgreen', lw=4)
        ]
        ax.set_xlim((0,1))
        ax.legend(custom_lines, [self.name1, self.name2], fontsize=20)
        return fig, (ax, axt)

        


if __name__ == '__main__':
    # Only for testing functionality
    data1 = '/Data/ML4Reco/test_newformat/dataCollection.djcdc'
    data2 = '/Data/ML4Reco/test_newformat_preselection/dataCollection.djcdc'
    model1 = '/Data/ML4Reco/standard_model/KERAS_check_model_block_5_epoch_50.h5'
    model2 = '/Data/ML4Reco/preselection_model/KERAS_check_model_block_5_epoch_150.h5'
    prediction1 = '/Data/ML4Reco/full_pred_202_nanoML.bin.gz'
    prediction2 = '/Data/ML4Reco/full_pred_202_nanoML.bin.gz'

    # cplot = ComparisonPlotter(model1=model1, model2=model2, dc1=data1, dc2=data2, Nfiles=1, Nsteps=1)
    cplot = ComparisonPlotter(prediction1=prediction1, prediction2=prediction2)

    fig, (ax1, ax2) = cplot.energy_distribution(title1='Some title', title2='Another title')
    fig, (ax1, ax2) = cplot.resolution_plots()
    fig, (ax, axt) = cplot.matched_showers()
    # plt.savefig('/Data/ML4Reco/Plots/resolutions.png')
    # plt.savefig('/Data/ML4Reco/Plots/showers.png')