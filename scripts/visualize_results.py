#!/usr/bin/env python3

import pdb
import os
import gzip
import mgzip
import pickle
import argparse
import time

import pandas as pd
import plotly.express as px

from ShowersMatcher import ShowersMatcher
from OCHits2Showers import OCHits2Showers


def to_plotly_true(feat_dict, truth_dict, file,  stringy=False, override_sids=None, range=None, color_scale=px.colors.sequential.Rainbow, add_in_color=0):
    print(truth_dict.keys())
    truth_assignment = truth_dict['truthHitAssignementIdx'][:,0]
    # truth_assignment = ['s'+str(x) if x==-1 else x for x in truth_assignment]
    # if stringy:
    #     truth_assignment = np.array(['s%d'%x for x in truth_assignment.tolist()])

    pdb.set_trace()
    result = {
        'x (cm)': feat_dict['recHitX'][:,0],
        'y (cm)': feat_dict['recHitY'][:,0],
        'z (cm)': feat_dict['recHitZ'][:,0],
        'rechit_energy': feat_dict['recHitEnergy'][:,0],
        'truth_assignment': truth_assignment if override_sids is None else override_sids,
        'truth_assignment_energy': truth_dict['truthHitAssignedEnergies'][:,0],
        'truth_assignment_energy_dep': truth_dict['t_rec_energy'][:,0],
    }

    if stringy:
        result['truth_assignment'] = ['s'+str(x) for x in (result['truth_assignment']+add_in_color)]

    result['size'] = np.log(result['rechit_energy']+1)
    # result['color'] = ['s'+str(x) for x in result['truth_assignment']]

    hover_data = [k for k,v in result.items()]
    df = pd.DataFrame(result)

    print("Uniques", np.unique(result['truth_assignment']))
    color_discrete_map = {
        's-1':'darkgray',
        's0':'blueviolet',
    }

    fig = px.scatter_3d(df, x="z (cm)", y="x (cm)", z="y (cm)",
                        color="truth_assignment", size="size",
                        # symbol="recHitID",
                        hover_data=hover_data,
                        template='plotly_dark' if file.endswith('.html') else 'ggplot2',
                        range_color=range,
                        # color_continuous_scale=color_scale,
                        color_discrete_map=color_discrete_map)

    if file.endswith('.html'):
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.write_html(file)
    else:
        tune_for_png(fig)
        fig.write_image(file,width=1500,height=1500)


def to_plotly_pred(feat_dict, pred_dict, file,  stringy=False, override_sids=None, range=None, color_scale=px.colors.sequential.Rainbow, add_in_color=0):
    print(pred_dict.keys())
    pred_assignment = pred_dict['pred_sid'][:,0]
    # truth_assignment = ['s'+str(x) if x==-1 else x for x in truth_assignment]
    # if stringy:
    #     truth_assignment = np.array(['s%d'%x for x in truth_assignment.tolist()])

    result = {
        'x (cm)': feat_dict['recHitX'][:,0],
        'y (cm)': feat_dict['recHitY'][:,0],
        'z (cm)': feat_dict['recHitZ'][:,0],
        'rechit_energy': feat_dict['recHitEnergy'][:,0],
        'pred_assignment': pred_assignment if override_sids is None else override_sids,
        'pred_assignment_energy': pred_dict['pred_energy'][:,0],
        # 'truth_assignment_energy_dep': truth_dict['t_rec_energy'][:,0],
    }

    if stringy:
        result['pred_assignment'] = ['s'+str(x) for x in (result['pred_assignment']+add_in_color)]


    result['size'] = np.log(result['rechit_energy']+1)
    # result['color'] = ['s'+str(x) for x in result['truth_assignment']]

    color_discrete_map = {
        's-1': 'darkgray',
        's0': 'blueviolet',
    }

    hover_data = [k for k,v in result.items()]
    df = pd.DataFrame(result)
    fig = px.scatter_3d(df, x="z (cm)", y="x (cm)", z="y (cm)",
                        color="pred_assignment", size="size",
                        # symbol="recHitID",
                        hover_data=hover_data,
                        template='plotly_dark' if file.endswith('.html') else 'ggplot2',
                        range_color=range,
                        # color_continuous_scale=color_scale,
                        color_discrete_map=color_discrete_map)

    fig.update_traces(marker=dict(line=dict(width=0)))

    if file.endswith('.html'):
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.write_html(file)
    else:
        tune_for_png(fig)
        fig.write_image(file,width=1500,height=1500)


def analyse(preddir, outpath, beta_threshold, distance_threshold, iou_threshold, matching_mode, analysisoutpath, nfiles,
            local_distance_scaling, is_soft, op, de_e_cut, angle_cut, kill_pu=True):
    hits2showers = OCHits2Showers(beta_threshold, distance_threshold, is_soft, local_distance_scaling, op=op)
    showers_matcher = ShowersMatcher(matching_mode, iou_threshold, de_e_cut, angle_cut)

    files_to_be_tested = [os.path.join(preddir, x) for x in os.listdir(preddir) if x.endswith('.bin.gz')]
    if nfiles!=-1:
        files_to_be_tested = files_to_be_tested[0:min(nfiles, len(files_to_be_tested))]

    showers_dataframe = pd.DataFrame()
    event_id = 0

    for i, file in enumerate(files_to_be_tested):
        print("Analysing file", i, file)
        with mgzip.open(file, 'rb') as f:
            file_data = pickle.load(f)
            for j, endcap_data in enumerate(file_data):
                print("Analysing endcap",j)
                stopwatch = time.time()
                features_dict, truth_dict, predictions_dict = endcap_data
                processed_pred_dict, pred_shower_alpha_idx = hits2showers.call(features_dict, predictions_dict)
                print('took',time.time()-stopwatch,'s for inference clustering')
                stopwatch = time.time()
                showers_matcher.set_inputs(
                    features_dict=features_dict,
                    truth_dict=truth_dict,
                    predictions_dict=processed_pred_dict,
                    pred_alpha_idx=pred_shower_alpha_idx
                )
                showers_matcher.process()
                print('took',time.time()-stopwatch,'s to match')
                stopwatch = time.time()
                f_true = os.path.join(outpath, file + '_truth.html')
                f_pred = os.path.join(outpath, file + '_pred.html')
                pdb.set_trace()
                to_plotly_true(
                    feat_dict=features_dict, 
                    truth_dict=truth_dict, 
                    file=f_true,  
                    stringy=False, 
                    override_sids=showers_matcher.truth_sid_matched, 
                    range=None, 
                    color_scale=px.colors.sequential.Rainbow,
                    add_in_color=0
                    )
                to_plotly_pred(
                    feat_dict=features_dict,
                    pred_dict=predictions_dict, 
                    file=f_pred,  
                    stringy=False, 
                    override_sids=showers_matcher.pred_sid_matched, 
                    range=None, 
                    color_scale=px.colors.sequential.Rainbow, 
                    add_in_color=0
                    )
                """
                if False:
                    # Probably can be removed
                    dataframe = showers_matcher.get_result_as_dataframe()
                    print('took',time.time()-stopwatch,'s to make data frame')
                    dataframe['event_id'] = event_id
                    event_id += 1
                    if kill_pu:
                        from globals import pu
                        if len(dataframe[dataframe['truthHitAssignementIdx']>=pu.t_idx_offset]):
                            print('\nWARNING REMOVING PU TRUTH MATCHED SHOWERS, HACK.\n')
                            dataframe = dataframe[dataframe['truthHitAssignementIdx']<pu.t_idx_offset]
                    showers_dataframe = pd.concat((showers_dataframe, dataframe))
                """

    # This is only to write to pdf files
    """
    if False:
        scalar_variables = {
            'beta_threshold': str(beta_threshold),
            'distance_threshold': str(distance_threshold),
            'iou_threshold': str(iou_threshold),
            'matching_mode': str(matching_mode),
            'is_soft': str(is_soft),
            'de_e_cut': str(de_e_cut),
            'angle_cut': str(angle_cut),
        }

        if len(analysisoutpath) > 0:
            analysis_data = {
                'showers_dataframe' : showers_dataframe,
                'events_dataframe' : None,
                'scalar_variables' : scalar_variables,
            }
            with gzip.open(analysisoutpath, 'wb') as f:
                print("Writing dataframes to pickled file",analysisoutpath)
                pickle.dump(analysis_data,f)

        if len(outpath)>0:
            plotter = HGCalAnalysisPlotter()
            plotter.set_data(showers_dataframe, None, '', outpath, scalar_variables=scalar_variables)
            plotter.process()
    """

if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(
        'Analyse predictions from object condensation and plot relevant results')
    parser.add_argument('input', help='Directory with input file')
    args = parser.parse_args()
    sm = ShowersMatcher(
        match_mode='iou_max',
        iou_threshold=0.5,
        de_e_cut=-1,
        angle_cut=0.5,
        )
    print("DONE")
    '''

    parser = argparse.ArgumentParser(
        'Analyse predictions from object condensation and plot relevant results')
    parser.add_argument('preddir',
                        help='Directory with .bin.gz files or a txt file with full paths of the bin gz files from the prediction.')
    parser.add_argument('-p',
                        help='Output directory for the final analysis html files (otherwise, it won\'t be produced)',
                        default='')
    parser.add_argument('-b', help='Beta threshold (default 0.1)', default='0.1')
    parser.add_argument('-d', help='Distance threshold (default 0.5)', default='0.5')
    parser.add_argument('-i', help='IOU threshold (default 0.1)', default='0.1')
    parser.add_argument('-m', help='Matching mode', default='iou_max')
    parser.add_argument('--analysisoutpath', help='Will dump analysis data to a file to remake plots without re-running everything.',
                        default='')
    parser.add_argument('--nfiles', help='Maximum number of files. -1 for everything in the preddir',
                        default=-1)
    parser.add_argument('--no_local_distance_scaling', help='With local distance scaling', action='store_true')
    parser.add_argument('--de_e_cut', help='dE/E threshold to allow match.', default=-1)
    parser.add_argument('--angle_cut', help='Angle cut for angle based matching', default=-1)
    parser.add_argument('--no_op', help='Use condensate op', action='store_true')
    parser.add_argument('--no_soft', help='Use condensate op', action='store_true')

    args = parser.parse_args()

    analyse(preddir=args.preddir, outpath=args.p, beta_threshold=float(args.b), distance_threshold=float(args.d),
            iou_threshold=float(args.i), matching_mode=args.m, analysisoutpath=args.analysisoutpath,
            nfiles=int(args.nfiles), local_distance_scaling=not args.no_local_distance_scaling,
            is_soft=not args.no_soft, op=not args.no_op, de_e_cut=float(args.de_e_cut), angle_cut=float(args.angle_cut))


