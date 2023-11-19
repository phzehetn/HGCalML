
from test_data import make_data
from accknn_op import AccumulateKnn, _AccumulateKnnGrad
import time

## just load it all once
#make the data
feat, rs, nidx, distsq = make_data(
        N_vtx=10, N_feat=10, N_coords=2,  K=10
    )
#one to load all ops
x, _ = AccumulateKnn(distsq, feat, nidx,
                      mean_and_max=True, force_tf=False)
# done loading

def test(N_vtx, N_feat, K):
    print('test for', N_vtx, 'vertices', N_feat, 'features', K, 'neighbours')

    feat, rs, nidx, distsq = make_data(
        N_vtx=N_vtx, N_feat=N_feat, N_coords=2,  K=K
    )
    
    t_0 = time.time()
    # TF baseline
    for _ in range(10):
        x, _ = AccumulateKnn(distsq, feat, nidx,
                      mean_and_max=True, force_tf=True)
    t_time = time.time() - t_0
    print('tensorflow took',t_time/10, 's')
    
    # custom
    t_0 = time.time()
    for _ in range(10):
        x, _ = AccumulateKnn(distsq, feat, nidx,
                      mean_and_max=True, force_tf=False)
    c_time = time.time() - t_0
    print('custom took',c_time/10, 's')

    print('reduction to', int( c_time*100./t_time ),'%')




def test_gradient(N_vtx, N_feat, K):
    print('test gradient for', N_vtx, 'vertices', N_feat, 'features', K, 'neighbours')
    feat, rs, nidx, distsq = make_data(
        N_vtx=N_vtx, N_feat=N_feat, N_coords=2,  K=K
    )
    #one to load all ops
    x, midx = AccumulateKnn(distsq, feat, nidx,
                      mean_and_max=True, force_tf=False)

    class dummy(object):
        def __init__(self):
            self.inputs=[]
            self.outputs=[]
    op = dummy()
    op.inputs = [distsq, feat,nidx]
    op.outputs = [x, midx]

    #run once
    _AccumulateKnnGrad(op, x, None)
    t_0 = time.time()
    for _ in range(10):
        _AccumulateKnnGrad(op, x, None)
    c_time = time.time() - t_0
    print('custom grad took',c_time/10, 's')

if True:
    test_gradient(200000, 32, 32)
    test_gradient(200000, 32, 64)
    test_gradient(200000, 32, 128)
    
    test_gradient(200000, 64, 32)
    test_gradient(200000, 64, 64)
    test_gradient(200000, 64, 128)
    test_gradient(200000, 64, 256)

if True:
    test(100000, 32, 16)
    test(100000, 64, 16)
    test(100000, 64, 32)
    test(100000, 64, 64)
    test(100000, 64, 96)
    test(100000, 32, 96)
    test(100000, 32, 64)
    test(100000, 128, 96)