#!/usr/bin/env python3
import pdb

import os
import time
import pickle
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

from DeepJetCore.modeltools import load_model
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator


parser = ArgumentParser()
parser.add_argument('input_model')
parser.add_argument('data_collection',
    help="Data collection file in djcdc format from which to pick files \
        to run inference on. You can use valsamples.djcdc in \
        training folder as a starter.")
parser.add_argument('--output_file', type=str, default=None,
    help="Path for output data to be stored, if None, data won't be stored")
parser.add_argument('--output_plot', type=str, default=None,
    help="Path for summary plot to be stored, if None, plot won't be stored")
parser.add_argument("--max_files", type=int, default=-1,
    help="Limit number of files")
parser.add_argument("--max_steps", type=int, default=-1,
    help="Maximum number of steps per file")
parser.add_argument("--extended", 
    help="Use extended format for files",
    action='store_true')
args = parser.parse_args()

if not os.path.exists(args.input_model):
    print("Input model ", args.input_model, " not found!")
    raise RuntimeError

if not os.path.exists(args.data_collection):
    print("Data collection ", args.data_collection, " not found!")
    raise RuntimeError

if os.path.exists(args.output_file):
    print("File already exists, please choose different output file!")
    raise RuntimeError

model = load_model(args.input_model)
dc = DataCollection(args.data_collection)
input_files = [dc.dataDir + f for f in dc.samples]
if args.max_files != -1 and args.max_files < len(input_files):
    input_files = input_files[:args.max_files]

times = []
nhits = []
event = []
eventID = 0

for infile in input_files:
    td = dc.dataclass()
    td.readFromFileBuffered(infile)
    gen = TrainDataGenerator()
    gen.setBatchSize(1)
    gen.setSquaredElementsLimit(False)
    gen.setSkipTooLargeBatches(False)
    gen.setBuffer(td)
    num_steps = gen.getNBatches()
    if args.max_steps != -1 and args.max_steps < num_steps:
        num_steps = args.max_steps
    generator = gen.feedNumpyData()

    for i in range(num_steps):
        print("Predicting step ", i, " of ", num_steps)
        data = next(generator)
        t0 = time.time()
        prediction = model(data[0])
        t1 = time.time()
        times.append(t1 - t0)
        nhits.append(data[0][0].shape[0])
        event.append(eventID)
        eventID += 1
        print(t1 - t0)
        print(data[0][0].shape[0])

times = np.array(times)
nhits = np.array(nhits)
event = np.array(event)
out_data = {
    'times': times,
    'nhits': nhits,
    'event': event,
}

if args.output_file is not None:
    with open(args.output_file, 'wb') as f:
        pickle.dump(out_data, f)

if args.output_plot is not None:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    ax.scatter(nhits, times)
    ax.set_xlabel('Number of hits', fontsize=20)
    ax.set_ylabel('CPU time [s] on Laptop', fontsize=20)
    ax.set_title("Inference time for model (cheplike)", fontsize=25)