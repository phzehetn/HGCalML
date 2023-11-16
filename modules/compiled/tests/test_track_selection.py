
from test_data import make_data
import numpy as np
import tensorflow as tf

from GravNetLayersRagged import SelectTracks, ScatterBackTracks

def test_track_selection():
    #make the data
    N_vtx = 11
    N_feat = 4
    N_coords = 3
    K = 10

    feat, rs, nidx, distsq = make_data(N_vtx, N_feat, N_coords, K, nrs=4)
    #create fake random track identifiers, 0 and 1 same length as vertices

    track_ids = tf.constant( np.random.randint(0,2, size=N_vtx), dtype=tf.float32)[...,tf.newaxis]

    print(track_ids, feat, rs)
    tfeat, tridx, rs = SelectTracks(return_rs=True)([track_ids, feat , rs])
    print('selected\n','tfeat',tfeat,'tridx',tridx,'rs',rs)

    #now scatter back
    feat = ScatterBackTracks()([track_ids,tfeat , tridx])
    print(feat)


test_track_selection()