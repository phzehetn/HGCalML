#!/usr/bin/env python3
# parse an input dataset and an input model

import pdb
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import setGPU
import pandas as pd
from DeepJetCore import DataCollection
from djcdata.dataPipeline import TrainDataGenerator
from DeepJetCore.modeltools import load_model
from GraphCondensationLayers import GraphCondensation, get_unique_masks
from DebugLayers import switch_off_debug_plots
from LossLayers import LossLayerBase
from MetricsLayers import MLBase
from tqdm import tqdm
import time
import numba


def time_function(func, *args, **kwargs):
    start = time.time()
    res = func(*args, **kwargs)
    end = time.time()
    #get function name from func object
    # print(func.__name__, 'time', end-start, 's')
    return res


def parse_args():
    parser = argparse.ArgumentParser(description='parse an input dataset and an input model')
    parser.add_argument('input_model', type=str, help='input model')
    parser.add_argument('input_data', type=str, help='input dataset')
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('--max_files', type=int, help='maximum number of files to go through', default=-1)
    return parser.parse_args()


def main():

    args = parse_args()

    if not os.path.isfile(args.input_model):
        print(f"ERROR: Input model {args.input_model} not found")
        raise FileNotFoundError 

    if not os.path.isfile(args.input_data):
        print(f"ERROR: Input data {args.input_data} not found")
        raise FileNotFoundError 

    if not os.path.exists(args.output_dir):
        print(f"Creating output directory {args.output_dir}")
        os.makedirs(args.output_dir)

    print(f"Loading model {args.input_model} and deactivating metrics")
    model = load_model(args.input_model)
    model = switch_off_debug_plots(model)
    for l in model.layers:
        if isinstance(l, LossLayerBase):
            print('deactivating layer',l.name)
            l.active=False
        if isinstance(l, MLBase):
            print('deactivating metrics layer',l.name)
            l.active=False
        l.trainable = False

    print(f"Setting up generator")

    dc = DataCollection(args.input_data)
    if args.max_files > 0:
        n_files = min(args.max_files, len(dc.samples))
    else:
        n_files = len(dc.samples)

    pbar = tqdm(total=n_files)
    for n_file in range(n_files):
        gen = TrainDataGenerator()
        gen.setBatchSize(1)
        gen.setSquaredElementsLimit(False)
        gen.setSkipTooLargeBatches(False)

        td = dc.dataclass()
        td.readFromFileBuffered(os.path.join(dc.dataDir, dc.samples[n_file]))
        gen.setBuffer(td)
        generator = gen.feedNumpyData()
        num_steps = gen.getNBatches()
        print(f"Num_steps: {num_steps}")

        all_data = []
        output = []
        out_truth = {}

        for i in range(num_steps):
            
            data = next(generator)[0]

            start = time.time()
            predictions_dict = model(data)
            truth_dict = td.createTruthDict(data)
            features_dict = td.createFeatureDict(data)
            end  = time.time()

            all_data.append([features_dict, truth_dict, predictions_dict])
            
            pbar.update(1/num_steps)
            pbar.set_postfix({'inference ': '{:.2f} s'.format(end-start)})

            #print inference time in tqdm as additional info in the progress bar, format the time to 2 decimal places
        td.writeOutPredictionDict(all_data, os.path.join(args.output_dir,  f"precluster_{str(n_file).zfill(4)}.bin.gz"))
            

    pbar.close()

    return

if __name__ == "__main__":
    main()


"""
#make all arrays in the dictionary to be one-dimensional. if more than one dim exists, add '_0', '_1', etc. to the key
flat_out = {}
for k in out_truth.keys():
    flat_out[k] = np.squeeze(np.concatenate([d[k] for d in all_data]))

if len(flat_out):
    print([(k, flat_out[k].shape) for k in flat_out.keys()  ])
    df = pd.DataFrame(flat_out)
    #remove all entries with truthHitAssignementIdx < 0
    df = df[df['truthHitAssignementIdx']>=0]
    
    #save the data frame to a reasonably small file
    df.to_hdf(os.path.join(args.output_dir, 'presel_tree.h5'), key='df', mode='w')

#load dataframe back
df = pd.read_hdf(os.path.join(args.output_dir, 'presel_tree.h5'))

# print all column names
print(df.columns)
'''
'truthHitAssignementIdx', 'truthHitAssignedEnergies',
       'truthHitAssignedX', 'truthHitAssignedY', 'truthHitAssignedZ',
       'truthHitAssignedEta', 'truthHitAssignedPhi', 'truthHitAssignedT',
       'truthHitAssignedPIDs', 'truthHitSpectatorFlag',
       'truthHitFullyContainedFlag', 't_rec_energy', 'sel'
'''

energy_cut = 40
#plot the number of hits for each selected or not selected object
for k in df.columns:
    if k in ['truthHitAssignedX', 'truthHitAssignedY', 'truthHitAssignedZ']:
        continue
    plt.figure()
    plt.hist(df[k][df['sel']==1], bins=31, histtype='step', label='selected', density=True)
    plt.hist(df[k][df['sel']==0], bins=31, histtype='step', label='not selected', density=True)
    plt.xlabel(k)
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, k+'.png'))
    plt.close()

    #now the same but only for high energies truthHitAssignedEnergies > 40
    plt.figure()
    plt.hist(df[k][(df['sel']==1) & (df['truthHitAssignedEnergies']>energy_cut)], bins=31, histtype='step', label='selected', density=True)
    plt.hist(df[k][(df['sel']==0) & (df['truthHitAssignedEnergies']>energy_cut)], bins=31, histtype='step', label='not selected', density=True)
    plt.xlabel(k)
    plt.legend()
    plt.yscale('log')
    plt.savefig(os.path.join(args.output_dir, k+'_high_energy.png'))
    plt.close()

# make the next plots log scale in x and y (and adjust the binnings accordingly)
#one explicit for t_rec_energy / truthHitAssignedEnergies
plt.figure()
plt.yscale('log')
plt.xscale('log')
bins = np.logspace(-2, 2, 31)
plt.hist(df['t_rec_energy'][df['sel']==1]/df['truthHitAssignedEnergies'][df['sel']==1], bins=bins, histtype='step', label='selected', density=True)
plt.hist(df['t_rec_energy'][df['sel']==0]/df['truthHitAssignedEnergies'][df['sel']==0], bins=bins, histtype='step', label='not selected', density=True)
plt.xlabel('t_rec_energy / truthHitAssignedEnergies')
plt.legend()
plt.savefig(os.path.join(args.output_dir, 't_rec_energy_div_truthHitAssignedEnergies.png'))
plt.close()

#and same for high energies above energy_cut
plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.hist(df['t_rec_energy'][(df['sel']==1) & (df['truthHitAssignedEnergies']>energy_cut)]/df['truthHitAssignedEnergies'][(df['sel']==1) & (df['truthHitAssignedEnergies']>energy_cut)], bins=bins, histtype='step', label='selected', density=True)
plt.hist(df['t_rec_energy'][(df['sel']==0) & (df['truthHitAssignedEnergies']>energy_cut)]/df['truthHitAssignedEnergies'][(df['sel']==0) & (df['truthHitAssignedEnergies']>energy_cut)], bins=bins, histtype='step', label='not selected', density=True)
plt.xlabel('t_rec_energy / truthHitAssignedEnergies')
plt.legend()
plt.savefig(os.path.join(args.output_dir, 't_rec_energy_div_truthHitAssignedEnergies_high_energy.png'))
plt.close()

#similar plot for n_hits, also double log scale
plt.figure()
plt.yscale('log')
plt.xscale('log')
bins = np.logspace(0, 3, 31)
plt.hist(df['n_hits'][df['sel']==1], bins=bins, histtype='step', label='selected', density=True)
plt.hist(df['n_hits'][df['sel']==0], bins=bins, histtype='step', label='not selected', density=True)
plt.xlabel('n_hits')
plt.legend()
plt.savefig(os.path.join(args.output_dir, 'n_hits_dlog.png'))
plt.close()

#same plot, but only for high energies
plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.hist(df['n_hits'][(df['sel']==1) & (df['truthHitAssignedEnergies']>energy_cut)], bins=bins, histtype='step', label='selected', density=True)
plt.hist(df['n_hits'][(df['sel']==0) & (df['truthHitAssignedEnergies']>energy_cut)], bins=bins, histtype='step', label='not selected', density=True)
plt.xlabel('n_hits')
plt.legend()
plt.savefig(os.path.join(args.output_dir, 'n_hits_dlog_high_energy.png'))
plt.close()

#same plot but for photons, truthHitAssignedPIDs == 22 +- eps
eps = 0.1
plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.hist(df['n_hits'][(df['sel']==1) & (np.abs(df['truthHitAssignedPIDs']-22)<eps)], bins=bins, histtype='step', label='selected', density=True)
plt.hist(df['n_hits'][(df['sel']==0) & (np.abs(df['truthHitAssignedPIDs']-22)<eps)], bins=bins, histtype='step', label='not selected', density=True)
plt.xlabel('n_hits')
plt.legend()
plt.savefig(os.path.join(args.output_dir, 'n_hits_dlog_photons.png'))
plt.close()

#same for high energy
plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.hist(df['n_hits'][(df['sel']==1) & (np.abs(df['truthHitAssignedPIDs']-22)<eps) & (df['truthHitAssignedEnergies']>energy_cut)], bins=bins, histtype='step', label='selected', density=True)
plt.hist(df['n_hits'][(df['sel']==0) & (np.abs(df['truthHitAssignedPIDs']-22)<eps) & (df['truthHitAssignedEnergies']>energy_cut)], bins=bins, histtype='step', label='not selected', density=True)
plt.xlabel('n_hits')
plt.legend()
plt.savefig(os.path.join(args.output_dir, 'n_hits_dlog_photons_high_energy.png'))
plt.close()

#plot truthHitAssignedEta for same photon selection
plt.figure()
plt.hist(df['truthHitAssignedEta'][(df['sel']==1) & (np.abs(df['truthHitAssignedPIDs']-22)<eps)], bins=31, histtype='step', label='selected', density=True)
plt.hist(df['truthHitAssignedEta'][(df['sel']==0) & (np.abs(df['truthHitAssignedPIDs']-22)<eps)], bins=31, histtype='step', label='not selected', density=True)
plt.xlabel('truthHitAssignedEta')
plt.legend()
plt.savefig(os.path.join(args.output_dir, 'truthHitAssignedEta_photons.png'))
plt.close()

#same for high energy
plt.figure()
plt.hist(df['truthHitAssignedEta'][(df['sel']==1) & (np.abs(df['truthHitAssignedPIDs']-22)<eps) & (df['truthHitAssignedEnergies']>energy_cut)], bins=31, histtype='step', label='selected', density=True)
plt.hist(df['truthHitAssignedEta'][(df['sel']==0) & (np.abs(df['truthHitAssignedPIDs']-22)<eps) & (df['truthHitAssignedEnergies']>energy_cut)], bins=31, histtype='step', label='not selected', density=True)
plt.xlabel('truthHitAssignedEta')
plt.legend()
plt.savefig(os.path.join(args.output_dir, 'truthHitAssignedEta_photons_high_energy.png'))
plt.close()

def efficiency_plot(df, var,max=None, nbins=11):
    '''
    plots the efficiency (selected / all) in bins of a given variable
    '''
    plt.figure()
    if max is None:
        max = df[var].max()
    bins = np.linspace(df[var].min(), max , nbins)
    centers = 0.5*(bins[1:]+bins[:-1])
    all_h, _ = np.histogram(df[var], bins=bins)
    sel_h, _ = np.histogram(df[var][df['sel']==1], bins=bins)
    eff = sel_h/all_h
    eff_err = np.sqrt(eff*(1-eff)/all_h)
    print(eff_err.max())
    plt.errorbar(centers, eff, yerr=eff_err, fmt='o')
    plt.xlabel(var)
    plt.ylabel('efficiency')
    plt.savefig(os.path.join(args.output_dir, 'efficiency_'+var+'.png'))
    plt.close()

efficiency_plot(df, 'truthHitAssignedEnergies', max=30)
efficiency_plot(df[df['truthHitAssignedEnergies']>20], 'truthHitAssignedEta', )
efficiency_plot(df[df['truthHitAssignedEnergies']>20], 'truthHitAssignedPhi')
efficiency_plot(df[df['truthHitAssignedEnergies']>20], 'truthHitAssignedX')
efficiency_plot(df[df['truthHitAssignedEnergies']>20], 'truthHitAssignedY')
efficiency_plot(df, 'n_hits', max=500)
efficiency_plot(df, 't_rec_energy', max=70)
"""
