import pdb
import os
import sys
import pickle
import gzip
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from visualize_event import djcdc_to_dataframe
import extra_plots as ep

# get environment variable 'HGCALML'
try:
    HGCALML = os.environ['HGCALML']
    # from DeepJetCore.DataCollection import DataCollection
    # from DeepJetCore.dataPipeline import TrainDataGenerator
    from DeepJetCore import DataCollection # for new DJC version
    from djcdata.dataPipeline import TrainDataGenerator
except KeyError:
    HGCALML = None
    print("HGCALML not set, relying on gzip/pickle")




if __name__ == '__main__':

    INPUTFILE = sys.argv[1]
    OUTPUTDIR = sys.argv[2]
    if len(sys.argv) > 3:
        SAMPLERATE = int(sys.argv[3])

    if HGCALML is None and INPUTFILE.endswith('.djcdc'):
        print("HGCALML not set, cannot work with .djcdc files")
        sys.exit(1)

    if INPUTFILE.endswith('.djcdc'):
        df = djcdc_to_dataframe(INPUTFILE, 10)
        df.to_hdf(os.path.join(OUTPUTDIR, 'df.h5'), key='df')
    else:
        print("Unknown file type")
        sys.exit(1)


