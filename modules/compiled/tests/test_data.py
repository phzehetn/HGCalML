###
#
# just a small file to create test data
#
###

import numpy as np
import tensorflow as tf
from binned_select_knn_op import BinnedSelectKnn

def make_data(N_vtx, N_feat, N_coords,  K=0, nrs = 2):
    # create random data
    feat = tf.constant(np.random.rand(N_vtx, N_feat), dtype=tf.float32)
    coords = tf.constant(np.random.rand(N_vtx, N_coords), dtype=tf.float32)
    if nrs == 2:
        rs = tf.constant([0, N_vtx//2, N_vtx], dtype=tf.int32)
    elif nrs == 3:
        rs = tf.constant([0, N_vtx//3, N_vtx//3+N_vtx//3, N_vtx], dtype=tf.int32)
    elif nrs == 4:
        rs = tf.constant([0, N_vtx//4, N_vtx//2, N_vtx//2 + N_vtx//4, N_vtx], dtype=tf.int32)
    else:
        rs = tf.constant([0, N_vtx], dtype=tf.int32)
    if K==0:
        return feat, rs
        
    nidx,distsq = BinnedSelectKnn(K, coords, rs)
    return feat, rs, nidx, distsq
