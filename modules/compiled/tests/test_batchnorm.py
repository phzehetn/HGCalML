
from GravNetLayersRagged import ScaledGooeyBatchNorm2 
from test_data import make_data


def test_batchnorm():
    #make data
    N_vtx = 2
    N_feat = 4
    N_coords = 3
    K = 10

    feat, rs, nidx, distsq = make_data(N_vtx, N_feat, N_coords, K, nrs=1)
    feat *= 100.
    # create layer
    layer = ScaledGooeyBatchNorm2(fluidity_decay=0.5, max_viscosity=0.1, no_gaus = True, trainable=False)

    for i in range(10):
        print(feat)
        feat = layer(feat)
        
        print(layer.viscosity)

test_batchnorm()