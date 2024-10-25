#!/usr/bin/env python3
"""
Analysis script to run of the predictions of the model.
"""

import pdb
import os
import argparse
import pickle
import gzip
import mgzip
import pandas as pd
import numpy as np
import tensorflow as tf
import awkward as ak
import fastjet
import vector
import time

from OCHits2Showers import OCHits2ShowersLayer
from process_endcap import process_endcap, OCGatherEnergyCorrFac
from ShowersMatcher3 import ShowersMatcher
from hplots.hgcal_analysis_plotter import HGCalAnalysisPlotter
import extra_plots as ep
from visualize_event import dataframe_to_plot, matched_plot


def filter_dictionary(dictionary, mask):
    """
    Apply a mask to all values of a dictionary where it matches the shape
    """
    n_mask = len(mask) 
    for key, value in dictionary.items():
        if value.shape[0] == n_mask:
            dictionary[key] = value[mask]

    return dictionary


def weighted_avg(group, col_name):
    """
    To be used together with 'get_showers' to weigh entries with energy in groupby objects
    """
    return (group[col_name] * group['recHitEnergy']).sum() / group['recHitEnergy'].sum()


def get_showers(df, ignore_tracks=True, ignore_PU=False):
    """
    For truth-based clustering
    df -> Needs features and truth entries as well as 'event_id'
    """
    showers = []
    for eid in df.event_id.unique():
        df_i = df[df.event_id == eid]
        if ignore_tracks:
            df_i = df_i[df_i.recHitID == 0]
        if ignore_PU:
            df_i = df_i[df_i.t_only_minbias == 0]
        shower = df_i.groupby('truthHitAssignementIdx').apply(
            lambda group: pd.Series({
                'energy_sum': group['recHitEnergy'].sum(),
                'energy_true': group['truthHitAssignedEnergies'].mean(),
                'x_weighted_avg': weighted_avg(group, 'recHitX'),
                'y_weighted_avg': weighted_avg(group, 'recHitY'),
                'z_weighted_avg': weighted_avg(group, 'recHitZ')
            })
        ).reset_index()
        shower['event_id'] = eid
        showers.append(shower)
    showers = pd.concat(showers, ignore_index=True)
    return showers


def pred_to_4momentum(df):
    """
    For showers coming from 'get_showers' (truth-based)
    Return 4-momentum as awkward arrays to be used by fastjet
    """
    pred_energy = df['energy_sum'].values
    pred_x = df['x_weighted_avg'].values
    pred_y = df['y_weighted_avg'].values
    pred_z = df['z_weighted_avg'].values
    pred_px = pred_x * pred_energy / np.sqrt(pred_x**2 + pred_y**2 + pred_z**2)
    pred_py = pred_y * pred_energy / np.sqrt(pred_x**2 + pred_y**2 + pred_z**2)
    pred_pz = pred_z * pred_energy / np.sqrt(pred_x**2 + pred_y**2 + pred_z**2)
    # use px, py, pz and e to create an awkard Momentum4D array
    pred_momentum = ak.zip(
        {
            "px": pred_px,
            "py": pred_py,
            "pz": pred_pz,
            "E": pred_energy,
        },
        with_name="Momentum4D",
    )
    return pred_momentum


def pred_to_4momentum_origin(showers, showertype="pred"):
    """
    For showers dataframe created directly from 
    Return 4-momentum as awkward arrays to be used by fastjet
    """
    if showertype.lower() == "pred":
        pred = showers[~np.isnan(showers['pred_energy'])]
        energy = pred['pred_energy'].values
        x = np.nan_to_num(pred['pred_mean_x'].values, nan=0.0)
        y = np.nan_to_num(pred['pred_mean_y'].values, nan=0.0)
        z = np.nan_to_num(pred['pred_mean_z'].values, nan=0.0)
    elif showertype.lower() == "truth":
        truth = showers[~np.isnan(showers['truthHitAssignedEnergies'])]
        energy = truth['truthHitAssignedEnergies']
        x = np.nan_to_num(truth['truth_mean_x'].values, nan=0.0)
        y = np.nan_to_num(truth['truth_mean_y'].values, nan=0.0)
        z = np.nan_to_num(truth['truth_mean_z'].values, nan=0.0)
    else:
        raise NotImplementedError
    px = np.nan_to_num(x * energy / np.sqrt(x**2 + y**2 + z**2), nan=0.0)
    py = np.nan_to_num(y * energy / np.sqrt(x**2 + y**2 + z**2), nan=0.0)
    pz = np.nan_to_num(z * energy / np.sqrt(x**2 + y**2 + z**2), nan=0.0)
    pred_momentum = ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": energy,
        },
        with_name="Momentum4D",
    )
    return pred_momentum


def awkwardjets_to_pandas(jets):
    """
    Turn awkward jets back to pandas dataframe for convenience
    """
    jets_df = pd.DataFrame()
    jets_df['px'] = [jet.px for jet in jets]
    jets_df['py'] = [jet.py for jet in jets]
    jets_df['pz'] = [jet.pz for jet in jets]
    jets_df['E'] = [jet.E for jet in jets]
    jets_df['pt'] = [jet.pt for jet in jets]
    jets_df['eta'] = [jet.eta for jet in jets]
    jets_df['theta'] = [jet.theta for jet in jets]
    jets_df['phi'] = [jet.phi for jet in jets]
    jets_df['mass'] = [jet.mass for jet in jets]
    jets_df['x_plot'] = jets_df['E'] * np.cos(jets_df['phi']) * np.sin(jets_df['theta'])
    jets_df['y_plot'] = jets_df['E'] * np.sin(jets_df['phi']) * np.sin(jets_df['theta'])
    jets_df['z_plot'] = jets_df['E'] * np.cos(jets_df['theta'])
    # sort by E
    jets_df = jets_df.sort_values(by=['E'], ascending=False)
    return jets_df


def analyse(preddir,
            analysisoutpath,
            beta_threshold,
            distance_threshold,
            iou_threshold,
            nfiles,
            nevents,
            minbias_threshold=-1.,
            jet_parameter=-1.,
        ):

    files_to_be_tested = [
        os.path.join(preddir, x)
        for x in os.listdir(preddir)
        if (x.endswith('.bin.gz') and x.startswith('pred'))]
    if nfiles!=-1:
        files_to_be_tested = files_to_be_tested[0:min(nfiles, len(files_to_be_tested))]
    print(f"Testing files {files_to_be_tested}\n")

    hits2showers = OCHits2ShowersLayer(
        beta_threshold,
        distance_threshold,
        use_local_distance_thresholding=True)
    showers_matcher = ShowersMatcher('iou_max', iou_threshold, de_e_cut=-1, angle_cut=-1, shower0=False)
    energy_gatherer = OCGatherEnergyCorrFac()
    print("CONFIGURATION:")
    print(f"Clustering: Basic\nBeta: {beta_threshold}\nDistance: {distance_threshold}")
    print(f"Using local distance scaling: True")
    print(f"Matching: IoU\nThreshold: {iou_threshold}")
    print(f"Using Energy Gatherer: {type(energy_gatherer).__name__}\n\n")

    ###############################################################################################
    ### Create Showers ############################################################################
    ###############################################################################################
    event_id = 0
    shower_dataframe = []
    for i, file in enumerate(files_to_be_tested):
        print(f"Analysing file {i+1}/{len(files_to_be_tested)}")
        with mgzip.open(file, 'rb') as f:
            file_data = pickle.load(f)
        for j, endcap_data in enumerate(file_data):
            if (nevents != -1) and (j > nevents):
                continue
            features_dict, truth_dict, predictions_dict = endcap_data

            if not int(args.m) == -1:
                minbias_mask = np.reshape(truth_dict['t_minbias_weighted'] <= args.m, (-1))
                features_dict = filter_dictionary(features_dict, minbias_mask)
                truth_dict = filter_dictionary(truth_dict, minbias_mask)
                predictions_dict = filter_dictionary(predictions_dict, minbias_mask)

            processed_pred_dict, pred_shower_alpha_idx = process_endcap(
                    hits2showers,
                    energy_gatherer,
                    features_dict,
                    predictions_dict,
                    is_minbias=np.zeros_like(features_dict['recHitEnergy'])
                    )

            showers_matcher.set_inputs(
                features_dict=features_dict,
                truth_dict=truth_dict,
                predictions_dict=processed_pred_dict,
                pred_alpha_idx=np.array(pred_shower_alpha_idx),
            )

            showers_matcher.process(extra=True)
            dataframe = showers_matcher.get_result_as_dataframe()
            dataframe['event_id'] = event_id
            shower_dataframe.append(dataframe)

            event_id += 1
    shower_dataframe = pd.concat(shower_dataframe)

    ###############################################################################################
    ### Create Jets 
    ###############################################################################################
    if (minbias_threshold != 1.0) and (jet_parameter != -1.):

        times_clustering = []
        vector.register_awkward()
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, jet_parameter)
        jets_truth = []
        jets_pred = []
        for event in shower_dataframe.event_id.unique():
            event_df = shower_dataframe[shower_dataframe.event_id == event]
            momentum_truth = pred_to_4momentum_origin(event_df, showertype='truth')
            momentum_pred = pred_to_4momentum_origin(event_df, showertype='pred')
            t0 = time.time()
            cluster_truth = fastjet.ClusterSequence(momentum_truth, jetdef)
            cluster_pred = fastjet.ClusterSequence(momentum_pred, jetdef)
            truth_jets = cluster_truth.inclusive_jets()
            truth_jets = awkwardjets_to_pandas(truth_jets)
            pred_jets = cluster_pred.inclusive_jets()
            pred_jets = awkwardjets_to_pandas(pred_jets)
            times_clustering.append(time.time() - t0)
            pred_jets['event_id'] = event
            truth_jets['event_id'] = event
            pred_jets['type'] = 'prediction'
            truth_jets['type'] = 'truth'
            jets_truth.append(truth_jets)
            jets_pred.append(pred_jets)
            
        print(f"Average time for fastjet Clustering: {np.mean(times_clustering)}")
        all_truth = pd.concat(jets_truth)
        all_pred = pd.concat(jets_pred)
        jets = pd.concat([all_truth, all_pred])
        jets.to_hdf('jets.h5', key='jets')
        # jet_data = {
            # "truth": pd.concat(jets_truth),
            # "prediction": pd.concat(jets_pred),
            # }
        # with gzip.open("jets.bin.gz", "wb") as fw:
            # pickle.dump(jet_data, fw)

        # Create truth-based jets

        # Create prediction-based jets

    ###############################################################################################
    ### Create output
    ###############################################################################################
    shower_dataframe.to_hdf(analysisoutpath, key='showers')

    print("DONE")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analyse predictions from object condensation')
    parser.add_argument('preddir', help='Directory with .bin.gz files')
    parser.add_argument('-o', help='Saves shower data frame to hdf format', default='showers.hdf')
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.34)', default='0.34')
    parser.add_argument('-m',
        help='Threshold to filter weighted minbias. 0.9 might be reasonable',
        default=-1., type=float)
    parser.add_argument('-R',
        help='Parameter for anti-kt jet-clustering (0.6 standard, default: -1 -> off',
        default=-1., type=float)
    parser.add_argument('--nfiles', help='Maximum number of files.', default=-1)
    parser.add_argument('--nevents', help='Maximum number of events (per file)', default=-1)
    args = parser.parse_args()

    if args.m != -1:
        assert args.m >= 0.0
        assert args.m <= 1.0
    if args.R != -1:
        assert args.R >= 0.0
        assert args.R <= 1.0

    analyse(preddir=args.preddir,
            analysisoutpath=args.o,
            beta_threshold=float(args.b),
            distance_threshold=float(args.d),
            iou_threshold=float(args.i),
            nfiles=int(args.nfiles),
            nevents=int(args.nevents),
            minbias_threshold=float(args.m),
            jet_parameter=float(args.R),
            )

