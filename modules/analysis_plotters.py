import pdb
import os
import re

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
                    pass
        else:
            raise ValueError
    return df


class ComparisonPlotter():

    def __init__(self, model1, model2, dc1, dc2=None):
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
        self.processed = False

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


    def process(self, Nfiles=1, nsteps=-1):
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
            if nsteps == -1:
                nsteps = min(num_steps1, num_steps2)

            for i in range(nsteps):
                print("Processing step ", str(i), " of ", str(nsteps))
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
                hits_df1 = pd.DataFrame()
                hits_df2 = pd.DataFrame()
                for d in [processed_pred_dict1, feat_dict1, truth_dict1]:
                    hits_df1 = add_dict_to_df(hits_df1, d)
                for d in [processed_pred_dict2, feat_dict2, truth_dict2]:
                    hits_df2 = add_dict_to_df(hits_df2, d)

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


    def energy_distribution(self, nbins=50, title1='', title2=''):
        rechits1 = self.all_hits1['recHitEnergy'].to_numpy()
        rechits2 = self.all_hits2['recHitEnergy'].to_numpy()
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
        ax2 = ax1.twinx()
        n, bins, patches = ax1.hist([rechits1, rechits2], bins=nbins)
        ax1.cla()
        width = 0.4 * (bins[1] - bins[0])
        bins_shifted = bins + width
        ax1.bar(bins[:-1], n[0], width, align='edge', color='darkslateblue', label='standard')
        ax2.bar(bins_shifted[:-1], n[1], width, align='edge', color='forestgreen', label='preselection')
        ax1.set_xlabel("Energy", fontsize=20)
        ax1.set_ylabel("Without preselection", color='darkslateblue', fontsize=20)
        ax1.tick_params(axis='y', labelcolor='darkslateblue')
        ax2.set_ylabel("With preselection", color='forestgreen', fontsize=20, labelpad=25., rotation=270)
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



if __name__ == '__main__':
    data1 = '/Data/ML4Reco/test_newformat/dataCollection.djcdc'
    data2 = '/Data/ML4Reco/test_newformat_preselection/dataCollection.djcdc'
    model1 = '/Data/ML4Reco/standard_model/KERAS_check_model_block_5_epoch_50.h5'
    model2 = '/Data/ML4Reco/preselection_model/KERAS_check_model_block_5_epoch_150.h5'

    cplot = ComparisonPlotter(model1=model1, model2=model2, dc1=data1, dc2=data2)
    cplot.process(Nfiles=1, nsteps=1)

    fig, (ax1, ax2) = cplot.energy_distribution(title1='Some title', title2='Another title')