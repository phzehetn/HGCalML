import pdb

import os
import sys
import gzip
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analysis')
    parser.add_argument("prediction", 
        help='Predictions file (ends with .bin.gz)'
        )
    parser.add_argument('output_dir',
        help='Directory where plots will be stored'
        )
    args = parser.parse_args()

    with gzip.open(args.prediction, 'rb') as f:
        prediction = pickle.load(f)