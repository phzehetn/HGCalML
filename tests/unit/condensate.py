
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from assign_condensate_op import calc_ragged_shower_indices
from assign_condensate_op import BuildAndAssignCondensatesBinned, calc_ragged_cond_indices
from RaggedLayers import RaggedMixHitAndCondInfo, RaggedSelectFromIndices, RaggedToFlatRS
from GravNetLayersRagged import RaggedGravNet


def make_data(nx, ny, nrs, span=6):

    nx, ny = (nx, ny)

    def oners():
        x = np.linspace(0, span, nx)
        y = np.linspace(0, span, ny)
        xv, yv = np.meshgrid(x, y)

        x = np.concatenate([xv[..., np.newaxis], yv[..., np.newaxis]], axis=-1)
        x = np.reshape(x, [nx * ny, 2])
        beta = np.random.rand(nx * ny, 1)

        return x, beta, nx * ny

    xs = []
    betas = []
    rs = [0]
    for _ in range(nrs):
        x, b, r = oners()
        xs.append(x)
        betas.append(b)
        rs.append(r)

    rs = np.cumsum(np.array(rs), axis=0)
    xs = np.concatenate(xs, axis=0)
    betas = np.concatenate(betas, axis=0)

    return tf.constant(xs, dtype='float32'), tf.constant(
        betas, dtype='float32'), tf.constant(rs, dtype='int32')


def test_indices(betathresh, keep_noise, weird_coordinates):
    x, b, rs = make_data(4, 3, 2, span=10)

    nocond = None  # tf.where( b <0.1, 1, tf.zeros_like(b))

    if weird_coordinates:
        x = 0. * x

    assignment, asso, alphaidx, is_cond, ncond = BuildAndAssignCondensatesBinned(
        x, b, tf.ones_like( b),
        rs, betathresh, no_condensation_mask=nocond,
        keep_noise=keep_noise, assign_by_max_beta=False)

    # ncondensates does not include noise, but then it is used as if it would
    # FIXME recalc on the fly using max and min assignment index per row split
    # FIXME!
    irdxs = calc_ragged_shower_indices(assignment, rs)

    #print('indices A done>>>', ncond, assignment)
    rcidx, retrev, flatrev = calc_ragged_cond_indices(
        assignment, alphaidx, ncond, rs)

    print('row_lengths', irdxs.row_lengths())

    c_x = RaggedSelectFromIndices()([x, rcidx])
    ch_x = RaggedSelectFromIndices()([x, irdxs])
    ch_x = RaggedMixHitAndCondInfo('add')([ch_x, c_x])

    xf, xfrs = RaggedToFlatRS()(c_x)

    print('>>>>>', xf.shape, xfrs[-1])
    xf, *_ = RaggedGravNet(
        n_neighbours=5, n_dimensions=3, n_filters=12, n_propagate=12
    )([xf, xfrs])

    # print(alphaidx)
    #print(xf.shape, xfrs, ncond, rcidx, irdxs.row_splits)
    tf.assert_equal(xfrs, irdxs.row_splits)

    # sanity check
    rassignment = tf.gather_nd(assignment, irdxs)
    rassignment = tf.reduce_min(rassignment, axis=2)  # same
    # rassignment = tf.ragged.boolean_mask(rassignment, rassignment <
    # 2147483647)#remove empty

    cprassignement = tf.gather_nd(assignment, rcidx)
    # first check
    #print('>>first', rassignment, '\n',cprassignement)
    tf.assert_equal((rassignment - cprassignement).values, 0, "first re-assign failed keep_noise: " + str(keep_noise) +
                    '\nrassignment' + str(rassignment) +
                    '\ncprassignement' + str(cprassignement)
                    )  # == for ragged sometimes creates issues

    # back from ragged
    retass = tf.gather_nd(cprassignement, retrev)
    #print('second >> ',retass, '\n',assignment)
    tf.assert_equal(
        retass,
        assignment,
        "second re-assign failed keep_noise: " +
        str(keep_noise))

    # back from flat
    #print('third a >> ',cprassignement, assignment, '\n',retrev, flatrev, ncond)
    retass = tf.gather_nd(cprassignement.values, flatrev)
    #print('third >> ',retass, '\n',assignment)
    tf.assert_equal(
        retass,
        assignment,
        "flat re-assign failed keep_noise: " +
        str(keep_noise))


def run_test():

    for i, bt in enumerate(10 * [0.00001, 0.01, 0.1, 0.2, 0.4, 0.9999]):
        np.random.seed(i + 1)
        print(bt)
        st = time.time()
        test_indices(bt, False, False)
        test_indices(bt, False, True)
        print('not keep noise', (time.time() - st) / 2.)
        st = time.time()
        np.random.seed(i + 1)
        test_indices(bt, True, False)
        test_indices(bt, True, True)
        print('keep noise', (time.time() - st) / 2.)
        # exit()


def test_plot():
    x, b, rs = make_data(80, 60, 2, span=10)

    assignment, asso, alphaidx, is_cond, ncond = BuildAndAssignCondensatesBinned(
        x, b, tf.ones_like( b), rs, 0.99,
        no_condensation_mask=None, keep_noise=True, assign_by_max_beta=False)

    irdxs = calc_ragged_shower_indices(assignment, rs)

    #print('indices A done>>>', ncond, assignment)
    rcidx, retrev, flatrev = calc_ragged_cond_indices(
        assignment, alphaidx, ncond, rs)

    # exit()
    xcp = tf.gather_nd(x, rcidx)
    xcpass = tf.gather_nd(assignment, rcidx)

    xr = tf.gather_nd(x, irdxs)
    xmr = tf.reduce_mean(xr, axis=2)

    '''
    make ragged condensation point select indices:
    event x CPs
    '''

    mrs = xmr.row_splits
    #print(xr, xmr, mrs)

    for i in range(len(rs) - 1):

        plt.scatter(x[rs[i]:rs[i + 1]][:, 0], x[rs[i]:rs[i + 1]]
                    [:, 1], c=assignment[rs[i]:rs[i + 1]], cmap='jet')

        plt.scatter(xmr[i, :, 0], xmr[i, :, 1], marker='x')
        plt.show()
        plt.close()


test_plot()
