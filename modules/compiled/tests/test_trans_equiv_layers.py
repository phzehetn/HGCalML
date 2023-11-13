
from GravNetLayersRagged import TranslationInvariantMP

from test_data import make_data

import tensorflow as tf
import numpy as np

#make data

N_vtx = 10
N_feat = 2
N_coords = 3
K = 2

feat, rs, nidx, distsq = make_data(N_vtx, N_feat, N_coords, K)

feat = feat*10.

#make layer
lay = TranslationInvariantMP([16])

#call layer
out = lay([feat, nidx, distsq])

feat = feat + 1000

out2 = lay([feat, nidx, distsq])

print(out-out2)