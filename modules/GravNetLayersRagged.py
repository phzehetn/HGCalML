import tensorflow as tf

from Initializers import EyeInitializer
from baseModules import LayerWithMetrics
from accknn_op import AccumulateKnn, AccumulateLinKnn
from binned_select_knn_op import BinnedSelectKnn
from neighbour_covariance_op import NeighbourCovariance as NeighbourCovarianceOp
from oc_helper_ops import SelectWithDefault
from slicing_knn_op import SlicingKnn


# First define simple helper functions


def layernorm(x, return_norm=False):
    # x = x - tf.reduce_mean(x,axis=-1, keepdims=True)
    norm = tf.reduce_sum(x**2, axis=-1, keepdims=True)
    norm = tf.sqrt(norm + 1e-6)
    corr = tf.sqrt(tf.cast(tf.shape(x)[-1], "float32"))
    if return_norm:
        x = tf.concat([x / norm * corr, norm], axis=-1)
    else:
        x = x / norm * corr
    return x


def AccumulateKnnSumw(distances, features, indices, mean_and_max=False):

    origshape = features.shape[1]
    features = tf.concat([features, tf.ones_like(features[:, 0:1])], axis=1)
    f, midx = AccumulateKnn(distances, features, indices, mean_and_max=mean_and_max)

    fmean = f[:, :origshape]
    fnorm = f[:, origshape : origshape + 1]
    fnorm = tf.where(fnorm < 1e-3, 1e-3, fnorm)
    fmean = tf.math.divide_no_nan(fmean, fnorm)
    fmean = tf.reshape(fmean, [-1, origshape])
    if mean_and_max:
        fmean = tf.concat([fmean, f[:, origshape + 1 : -1]], axis=1)
    return fmean, midx


def AccumulateLinKnnSumw(weights, features, indices, mean_and_max=False):

    origshape = features.shape[1]
    features = tf.concat([features, tf.ones_like(features[:, 0:1])], axis=1)
    f, midx = AccumulateLinKnn(weights, features, indices, mean_and_max=mean_and_max)

    fmean = f[:, :origshape]
    fnorm = f[:, origshape : origshape + 1]
    fmean = tf.math.divide_no_nan(fmean, fnorm)
    fmean = tf.reshape(fmean, [-1, origshape])
    if mean_and_max:
        fmean = tf.concat([fmean, f[:, origshape + 1 : -1]], axis=1)
    return fmean, midx


def check_type_return_shape(s):
    if not isinstance(s, tf.TensorSpec):
        raise TypeError(
            "Only TensorSpec signature types are supported, "
            "but saw signature entry: {}.".format(s)
        )
    return s.shape


# Layers are defined below


class CastRowSplits(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        This layer casts the row splits as they come from the data (int64, N_RS x 1)
        to (int32, N_RS), which is needed for subsequent processing. That's all
        """
        super(CastRowSplits, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CastRowSplits, self).build(input_shape)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0],)

    def call(self, inputs):
        assert inputs.dtype == "int64" or inputs.dtype == "int32"
        if len(inputs.shape) == 2:
            return tf.cast(inputs[:, 0], dtype="int32")
        elif inputs.dtype == "int64":
            return tf.cast(inputs, dtype="int32")
        else:
            return inputs


class Where(tf.keras.layers.Layer):
    def __init__(self, outputval=None, condition=">0", **kwargs):
        """
        Simple wrapper around tf.where.

        Inputs if outputval=None:
        - tensor defining condition
        - value to return if condition == True
        - value to return else

        Inputs if outputval=val:
        - tensor defining condition
        - value to return if condition is not fulfilled
         --> will return constant outputval=val if condition is fulfilled

        """
        conditions = [">0", ">=0", "<0", "<=0", "==0", "!=0"]
        assert condition in conditions
        self.condition = condition
        self.outputval = outputval

        super(Where, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(Where, self).get_config()
        return dict(
            list(base_config.items())
            + list({"outputval": self.outputval, "condition": self.condition}.items())
        )

    def build(self, input_shape):
        super(Where, self).build(input_shape)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1],)

    def call(self, inputs):

        if self.outputval is not None:
            assert len(inputs) == 2
            left = self.outputval
            right = inputs[1]
        else:
            assert len(inputs) == 3
            left = inputs[1]
            right = inputs[2]

        izero = tf.constant(0, dtype=inputs[0].dtype)
        if self.condition == ">0":
            return tf.where(inputs[0] > izero, left, right)
        elif self.condition == ">=0":
            return tf.where(inputs[0] >= izero, left, right)
        elif self.condition == "<0":
            return tf.where(inputs[0] < izero, left, right)
        elif self.condition == "<=0":
            return tf.where(inputs[0] <= izero, left, right)
        elif self.condition == "!=0":
            return tf.where(inputs[0] != izero, left, right)
        else:
            return tf.where(inputs[0] == izero, left, right)


############# Some layers for convenience ############


class PrintMeanAndStd(tf.keras.layers.Layer):
    def __init__(self, print_mean=True, print_std=True, print_shape=True, **kwargs):
        super(PrintMeanAndStd, self).__init__(**kwargs)
        self.print_mean = print_mean
        self.print_std = print_std
        self.print_shape = print_shape

    def get_config(self):
        config = {
            "print_mean": self.print_mean,
            "print_std": self.print_std,
            "print_shape": self.print_shape,
        }
        base_config = super(PrintMeanAndStd, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        # return input_shapes[0]
        return input_shapes

    def call(self, inputs):
        try:
            if self.print_mean:
                tf.print(self.name, "mean", tf.reduce_mean(inputs), summarize=100)
            if self.print_std:
                tf.print(self.name, "std", tf.math.reduce_std(inputs), summarize=100)
            if self.print_shape:
                tf.print(self.name, "shape", inputs.shape, summarize=100)
        except Exception:
            print("exception in", self.name)
            # raise e
        return inputs


class ProcessFeatures(tf.keras.layers.Layer):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        """
        'recHitEnergy': feat[:,0:1] ,          #recHitEnergy,
        'recHitEta'   : feat[:,1:2] ,          #recHitEta   ,
        'recHitID'    : feat[:,2:3] ,          #recHitID, #indicator if it is track or not
        'recHitTheta' : feat[:,3:4] ,          #recHitTheta ,
        'recHitR'     : feat[:,4:5] ,          #recHitR   ,
        'recHitX'     : feat[:,5:6] ,          #recHitX     ,
        'recHitY'     : feat[:,6:7] ,          #recHitY     ,
        'recHitZ'     : feat[:,7:8] ,          #recHitZ     ,
        'recHitTime'  : feat[:,8:9] ,            #recHitTime  
        'recHitHitR'  : feat[:,9:10] ,   
        """

        self.mean_hit = tf.constant(
            [
                0.0475,  # recHitEnergy
                2.55,  # recHitEta
                0.0,  # recHitID -> don't normalize
                0.167,  # recHitTheta
                341.4,  # recHitR
                0.0,  # recHitX -> centered around zero
                0.0,  # recHitY -> centered around zero
                336.0,  # recHitZ
                0.0,  # recHitTime -> All zeros
                0.95,  # recHitHitR
            ]
        )
        self.std_hit = tf.constant(
            [
                0.1991,  # recHitEnergy
                0.35,  # recHitEta
                1.0,  # recHitID -> don't normalize
                0.067,  # recHitTheta
                15.1,  # recHitR
                42.0,  # recHitX
                42.0,  # recHitY
                14.5,  # recHitZ
                1.0,  # recHitTime -> All zeros
                0.39,  # recHitHitR
            ]
        )

        self.mean_track = tf.constant(
            [
                3.04,  # recHitEnergy
                # the next seven are set to be the same as the hits on purpose
                2.55,  # recHitEta
                0.0,  # recHitID -> don't normalize
                0.167,  # recHitTheta
                341.4,  # recHitR
                0.0,  # recHitX -> centered around zero
                0.0,  # recHitY -> centered around zero
                336.0,  # recHitZ
                0.0,  # recHitTime -> All zeros
                0.0,  # recHitHitR -> All zeros for tracks
            ]
        )
        self.std_track = tf.constant(
            [
                3.63,  # recHitEnergy
                # the next seven are set to be the same as the hits on purpose
                0.35,  # recHitEta
                1.0,  # recHitID -> don't normalize
                0.067,  # recHitTheta
                15.1,  # recHitR
                42.0,  # recHitX
                42.0,  # recHitY
                14.5,  # recHitZ
                1.0,  # recHitTime -> All zeros
                1.0,  # recHitHitR -> All zeros for tracks
            ]
        )

    def get_config(self):
        config = {}
        base_config = super(ProcessFeatures, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        features = inputs
        is_track = tf.cast(
            features[:, 2:3], bool
        )  # is True if feature is != 0 (also true for -1)

        normalized_hits = (features - self.mean_hit) / self.std_hit
        normalized_tracks = (features - self.mean_track) / self.std_track
        normalized = tf.where(is_track, normalized_tracks, normalized_hits)

        return normalized


class NeighbourCovariance(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        Inputs:
          - coordinates (Vin x C)
          - distance squared (Vout x K)
          - features (Vin x F)
          - neighbour indices (Vout x K)

        Returns concatenated  (Vout x { F*C^2 + F*C})
          - feature weighted covariance matrices (lower triangle) (Vout x F*C^2)
          - feature weighted means (Vout x F*C)

        """
        super(NeighbourCovariance, self).__init__(**kwargs)
        self.outshapes = None

    def build(self, input_shapes):  # pure python
        super(NeighbourCovariance, self).build(input_shapes)

    @staticmethod
    def raw_get_cov_shapes(input_shapes):
        coordinates_s, features_s = None, None
        if len(input_shapes) == 4:
            coordinates_s, _, features_s, _ = input_shapes
        else:
            coordinates_s, features_s, _ = input_shapes
        nF = features_s[1]
        nC = coordinates_s[1]
        covshape = nF * nC**2  # int(nF*(nC*(nC+1)//2))
        return nF, nC, covshape

    @staticmethod
    def raw_call(coordinates, distsq, features, n_idxs):
        cov, means = NeighbourCovarianceOp(
            coordinates=coordinates,
            distsq=10.0 * distsq,  # same as gravnet scaling
            features=features,
            n_idxs=n_idxs,
        )
        return cov, means

    def call(self, inputs):
        coordinates, distsq, features, n_idxs = None, None, None, None
        if len(inputs) == 4:
            coordinates, distsq, features, n_idxs = inputs
        else:
            coordinates, features, n_idxs = inputs
            distsq = tf.zeros_like(n_idxs, dtype="float32")

        cov, means = NeighbourCovariance.raw_call(coordinates, distsq, features, n_idxs)

        nF, nC, covshape = NeighbourCovariance.raw_get_cov_shapes(
            [s.shape for s in inputs]
        )

        cov = tf.reshape(cov, [-1, covshape])
        means = tf.reshape(means, [-1, nF * nC])

        return tf.concat([cov, means], axis=-1)


class LocalDistanceScaling(LayerWithMetrics):
    def __init__(self, max_scale=10, **kwargs):
        """
        Inputs:
        - distances (V x N)
        - scaling (V x 1): (linear activation)

        Returns:
        distances * scaling : V x N x 1
        scaling is bound to be within 1/max_scale and max_scale.
        """
        super(LocalDistanceScaling, self).__init__(**kwargs)
        self.max_scale = float(max_scale)
        # some helpers
        self.c = 1.0 - 1.0 / max_scale
        self.b = max_scale / (2.0 * self.c)

        # derivative sigmoid = sigmoid(x) (1- sigmoid(x))
        # der sig|x=0 = 1/4
        # derivate of sigmoid (ax) = a (der sig)(ax)
        #

    def get_config(self):
        config = {"max_scale": self.max_scale}
        base_config = super(LocalDistanceScaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return input_shapes[1]

    @staticmethod
    def raw_call(dist, scale, a, b, c):
        # the derivative is continuous at 0
        scale_pos = a * tf.math.sigmoid(scale) + 1.0 - a / 2.0
        scale_neg = 2.0 * c * tf.math.sigmoid(b * scale) + 1.0 - c
        scale = tf.where(scale >= 0, scale_pos, scale_neg)
        return dist * scale, scale

    def call(self, inputs):
        dist, scale = inputs
        newdist, scale = LocalDistanceScaling.raw_call(
            dist, scale, self.max_scale, self.b, self.c
        )
        # self.add_prompt_metric(tf.reduce_mean(scale), self.name+'_dist_scale')
        # self.add_prompt_metric(tf.math.reduce_std(scale), self.name+'_var_dist_scale')
        return newdist


class SelectFromIndices(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """

        This layer selects a set of vertices.

        Inputs are:
         - the selection indices
         - a list of tensors the selection should be applied to (extending the indices)

        This layer is useful in combination with e.g. LocalClustering, to apply the clustering selection
        to other tensors (e.g. the output of a GravNet layer, or a SoftPixelCNN layer)


        """
        if "dynamic" in kwargs:
            super(SelectFromIndices, self).__init__(**kwargs)
        else:
            super(SelectFromIndices, self).__init__(dynamic=False, **kwargs)

    def compute_output_shape(self, input_shapes):  # these are tensors shapes
        # ts = tf.python.framework.tensor_shape.TensorShape
        outshapes = [(None,) + tuple(s[1:]) for s in input_shapes][1:]
        return outshapes  # all but first (indices)

    def compute_output_signature(self, input_signature):
        print(">>>>>SelectFromIndices input_signature", input_signature)
        input_shape = tf.nest.map_structure(check_type_return_shape, input_signature)
        output_shape = self.compute_output_shape(input_shape)
        input_dtypes = [i.dtype for i in input_signature]
        return [
            tf.TensorSpec(dtype=input_dtypes[i + 1], shape=output_shape[i])
            for i in range(0, len(output_shape))
        ]

    def build(self, input_shapes):  # pure python
        super(SelectFromIndices, self).build(input_shapes)
        outshapes = self.compute_output_shape(input_shapes)

        self.outshapes = [
            [
                -1,
            ]
            + list(s[1:])
            for s in outshapes
        ]

    @staticmethod
    def raw_call(indices, inputs, outshapes=None):
        if outshapes is None:
            outshapes = [
                [
                    -1,
                ]
                + list(s.shape[1:])
                for s in inputs
            ]
        outs = []
        for i in range(0, len(inputs)):
            g = tf.gather_nd(inputs[i], indices)
            g = tf.reshape(g, outshapes[i])  # [-1]+inputs[i].shape[1:])
            outs.append(g)
        if len(outs) == 1:
            return outs[0]
        return outs

    def call(self, inputs):
        indices = inputs[0]
        outshapes = (
            self.outshapes
        )  # self.compute_output_shape([tf.shape(i) for i in inputs])
        return SelectFromIndices.raw_call(indices, inputs[1:], outshapes)


class MultiBackGather(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """

        This layer gathers back vertices that were previously clustered using the output of LocalClustering.
        E.g. if vertices 0,1,2, and 3 ended up in the same cluster with 0 being the cluster centre, and 4,5,6 ended
        up in a cluster with 4 being the cluster centre, there will be two clusters: A and B.
        This layer will create a vector of the previous dimensionality containing:
        [A,A,A,A,B,B,B] so that the cluster properties of A and B are gathered back to the positions of their
        constituents.

        If multiple clusterings were performed, the layer can operate on a list of backgather indices.
        (internally the order of this list will be inverted)

        Inputs are:
         - The data to gather back to larger dimensionality by repetition
         - A list of backgather indices (one index tensor for each repetition).

        """
        if "dynamic" in kwargs:
            super(MultiBackGather, self).__init__(**kwargs)
        else:
            super(MultiBackGather, self).__init__(dynamic=False, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape  # batch dim is None anyway

    @staticmethod
    def raw_call(x, gathers):
        for k in range(len(gathers)):
            l = len(gathers) - k - 1
            # cast is needed because keras layer out dtypes are not really working
            x = SelectFromIndices.raw_call(
                tf.cast(gathers[l], tf.int32), [x], [[-1] + list(x.shape[1:])]
            )[0]
        return x

    def call(self, inputs):
        x, gathers = inputs
        return MultiBackGather.raw_call(x, gathers)


class MultiBackScatter(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """

        This layer scatters back vertices that were previously clustered using the output of LocalClustering.
        E.g. if vertices 0,1,2, and 3 ended up in the same cluster with 0 being the cluster centre, and 4,5,6 ended
        up in a cluster with 4 being the cluster centre, there will be two clusters: A and B.
        This layer will create a vector of the previous dimensionality containing:
        [A,0,0,0,B,0,0] so that the cluster properties of A and B are scattered back to the positions of their
        initial points.

        If multiple clusterings were performed, the layer can operate on a list of scatter indices.
        (internally the order of this list will be inverted)

        Inputs are:
         - The data to scatter back to larger dimensionality by repetition
         - A list of list of type: [backscatterV, backscatter indices ] (one index tensor for each repetition).

        """
        if "dynamic" in kwargs:
            super(MultiBackScatter, self).__init__(**kwargs)
        else:
            super(MultiBackScatter, self).__init__(dynamic=False, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape  # batch dim is None anyway

    @staticmethod
    def raw_call(x, scatters):
        xin = x
        for k in range(len(scatters)):
            l = len(scatters) - k - 1
            V, scidx = scatters[l]
            # print('scatters[l]',scatters[l])
            # cast is needed because keras layer out dtypes are not really working
            shape = tf.concat([tf.expand_dims(V, axis=0), tf.shape(x)[1:]], axis=0)
            x = tf.scatter_nd(scidx, x, shape)

        return tf.reshape(x, [-1, xin.shape[1]])

    def call(self, inputs):
        x, scatters = inputs
        if x.shape[0] is None:
            return tf.reshape(x, [-1, x.shape[1]])
        xnew = MultiBackScatter.raw_call(x, scatters)
        return xnew


class KNN(LayerWithMetrics):
    def __init__(
        self,
        K: int,
        radius=-1.0,
        use_approximate_knn=False,
        min_bins=None,
        tf_distance=False,
        **kwargs,
    ):
        """

        Select self+K nearest neighbours, with possible radius constraint.

        Call will return
         - self + K neighbour indices of K neighbours within max radius
         - distances to self+K neighbours

        Inputs: coordinates, row_splits

        :param K: number of nearest neighbours
        :param radius: maximum distance of nearest neighbours,
                       can also contain the keyword 'dynamic'
        :param use_approximate_knn: use approximate kNN method (SlicingKnn) instead of exact method (SelectKnn)
        """
        super(KNN, self).__init__(**kwargs)
        self.K = K

        self.use_approximate_knn = use_approximate_knn
        self.min_bins = min_bins
        self.tf_distance = tf_distance

        if isinstance(radius, int):
            radius = float(radius)
        self.radius = radius
        assert (isinstance(radius, str) and radius == "dynamic") or isinstance(
            radius, float
        )
        assert not (radius == "dynamic" and not use_approximate_knn)
        self.dynamic_radius = None
        if radius == "dynamic":
            radius = 1.0
            with tf.name_scope(self.name + "/1/"):
                self.dynamic_radius = tf.Variable(
                    initial_value=radius, trainable=False, dtype="float32"
                )

    def get_config(self):
        config = {
            "K": self.K,
            "radius": self.radius,
            "min_bins": self.min_bins,
            "use_approximate_knn": self.use_approximate_knn,
            "tf_distance": self.tf_distance,
        }
        base_config = super(KNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return (None, self.K + 1), (None, self.K + 1)

    @staticmethod
    def raw_call(
        coordinates,
        row_splits,
        K,
        radius,
        use_approximate_knn,
        min_bins,
        tfdist,
        myself,
    ):
        nbins = None
        if use_approximate_knn:
            if min_bins is None:
                min_bins = 11
            bin_width = radius  # default value for SlicingKnn kernel
            idx, dist, nbins = SlicingKnn(
                K + 1,
                coordinates,
                row_splits,
                features_to_bin_on=(0, 1),
                bin_width=(bin_width, bin_width),
                return_n_bins=True,
                min_bins=min_bins,
            )
        else:
            idx, dist = BinnedSelectKnn(
                K + 1,
                coordinates,
                row_splits,
                n_bins=min_bins,
                max_radius=radius,
                tf_compatible=False,
                name=myself.name,
            )

        if tfdist:
            ncoords = SelectWithDefault(idx, coordinates, 0.0)
            distsq = tf.reduce_sum((ncoords[:, 0:1, :] - ncoords) ** 2, axis=-1)
            distsq = tf.where(idx < 0, 0.0, distsq)

        idx = tf.reshape(idx, [-1, K + 1])
        dist = tf.reshape(dist, [-1, K + 1])

        return idx, dist, nbins

    def update_dynamic_radius(self, dist, training=None):
        if self.dynamic_radius is None or not self.trainable:
            return
        # update slowly, with safety margin
        update = (
            tf.math.reduce_max(tf.sqrt(dist)) * 1.05
        )  # can be inverted for performance TBI
        update = self.dynamic_radius + 0.1 * (update - self.dynamic_radius)
        updated_radius = tf.keras.backend.in_train_phase(
            update, self.dynamic_radius, training=training
        )
        tf.keras.backend.update(self.dynamic_radius, updated_radius)

    def call(self, inputs, training=None):
        coordinates, row_splits = inputs

        idx, dist = None, None
        if self.dynamic_radius is None:
            idx, dist, nbins = KNN.raw_call(
                coordinates,
                row_splits,
                self.K,
                self.radius,
                self.use_approximate_knn,
                self.min_bins,
                self.tf_distance,
                self,
            )
        else:
            idx, dist, nbins = KNN.raw_call(
                coordinates,
                row_splits,
                self.K,
                self.dynamic_radius,
                self.use_approximate_knn,
                self.min_bins,
                self.tf_distance,
                self,
            )
            self.update_dynamic_radius(dist, training)

        if self.use_approximate_knn:
            self.add_prompt_metric(nbins, self.name + "_slicing_bins")

        return idx, dist


class SortAndSelectNeighbours(tf.keras.layers.Layer):
    def __init__(
        self, K: int, radius: float = -1.0, sort=True, descending=False, **kwargs
    ):
        """

        This layer will sort neighbour indices by distance and possibly select neighbours
        within a radius, or the closest ones up to K neighbours.

        If a sorting score is given the sorting will be based on that (will still return the same)

        Inputs: distances, neighbour indices, sorting_score (opt)

        Call will return
         - neighbour distances sorted by distance
         - neighbour indices sorted by distance


        :param K: number of nearest neighbours, will do no selection if K<1
        :param radius: maximum distance of nearest neighbours (no effect if < 0)
        :param descending: use descending order

        """
        super(SortAndSelectNeighbours, self).__init__(**kwargs)
        self.K = K
        self.radius = radius
        self.sort = sort
        self.descending = descending

    def get_config(self):
        config = {
            "K": self.K,
            "radius": self.radius,
            "sort": self.sort,
            "descending": self.descending,
        }
        base_config = super(SortAndSelectNeighbours, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        if self.K > 0:
            return (None, self.K), (None, self.K)
        else:
            return input_shapes

    def compute_output_signature(self, input_signature):

        input_shapes = [x.shape for x in input_signature]
        input_dtypes = [x.dtype for x in input_signature]
        output_shapes = self.compute_output_shape(input_shapes)

        return [
            tf.TensorSpec(dtype=input_dtypes[i], shape=output_shapes[i])
            for i in range(len(output_shapes))
        ]

    @staticmethod
    def raw_call(
        distances,
        nidx,
        K,
        radius=-1,
        sort=True,
        incr_sorting_score=None,
        keep_self=True,
    ):

        K = K if K > 0 else distances.shape[1]
        if not sort:
            return distances[:, :K], nidx[:, :K]

        if incr_sorting_score is None:
            incr_sorting_score = distances
        elif (
            tf.shape(incr_sorting_score)[1] is not None
            and tf.shape(incr_sorting_score)[1] == 1
        ):
            incr_sorting_score = SelectWithDefault(nidx, incr_sorting_score, 0.0)[:, 0]

        tfssc = tf.where(
            nidx < 0, 1e9, incr_sorting_score
        )  # make sure the -1 end up at the end
        if keep_self:
            tfssc = tf.concat(
                [
                    tf.reduce_min(tfssc[:, 1:], axis=1, keepdims=True) - 1.0,
                    tfssc[:, 1:],
                ],
                axis=1,
            )  # make sure 'self' remains in first place

        sorting = tf.argsort(tfssc, axis=1)

        snidx = tf.gather(nidx, sorting, batch_dims=1)  # _nd(nidx,sorting,batch_dims=1)
        sdist = tf.gather(distances, sorting, batch_dims=1)
        if K > 0:
            snidx = snidx[:, :K]
            sdist = sdist[:, :K]

        if radius > 0:
            snidx = tf.where(sdist > radius, -1, snidx)
            sdist = tf.where(sdist > radius, 0.0, sdist)

        # fix the shapes
        sdist = tf.reshape(sdist, [-1, K])
        snidx = tf.reshape(snidx, [-1, K])

        return sdist, tf.cast(
            snidx, tf.int32
        )  # just to avoid keras not knowing the dtype

    def call(self, inputs):
        distances, nidx, ssc = None, None, None
        if len(inputs) == 2:
            distances, nidx = inputs
            ssc = distances

        elif len(inputs) == 3:
            distances, nidx, ssc = inputs

        if self.descending:
            ssc = -1.0 * ssc

        return SortAndSelectNeighbours.raw_call(
            distances, nidx, self.K, self.radius, self.sort, ssc
        )
        # make TF compatible


class RaggedGravNet(tf.keras.layers.Layer):
    def __init__(
        self,
        n_neighbours: int,
        n_dimensions: int,
        n_filters: int,
        n_propagate: int,
        return_self=True,
        sumwnorm=False,
        feature_activation="relu",
        use_approximate_knn=False,
        coord_initialiser_noise=1e-2,
        use_dynamic_knn=True,
        debug=False,
        n_knn_bins=None,
        _promptnames=None,  # compatibility, does nothing
        record_metrics=False,  # compatibility, does nothing
        **kwargs,
    ):
        """
        Call will return output features, coordinates, neighbor indices and squared distances from neighbors
        Inputs:
        - features
        - row splits

        :param n_neighbours: neighbors to do gravnet pass over
        :param n_dimensions: number of dimensions in spatial transformations
        :param n_filters:  number of dimensions in output feature transformation, could be list if multiple output
        features transformations (minimum 1)

        :param n_propagate: how much to propagate in feature tranformation, could be a list in case of multiple
        :param return_self: for the neighbour indices and distances, switch whether to return the 'self' index and distance (0)
        :param sumwnorm: normalise distance weights such that their sum is 1. (default False)
        :param feature_activation: activation to be applied to feature creation (F_LR) (default relu)
        :param use_approximate_knn: use approximate kNN method (SlicingKnn) instead of exact method (SelectKnn)
        :param use_dynamic_knn: uses dynamic adjustment of kNN binning derived from previous batches (only in effect together with use_approximate_knn)
        :param debug: switches on debug output, is not persistent when saving
        :param n_knn_bins: number of bins for included kNN (default: None=dynamic)
        :param kwargs:
        """
        super(RaggedGravNet, self).__init__(**kwargs)

        # n_neighbours += 1  # includes the 'self' vertex
        assert n_neighbours > 1
        assert not use_approximate_knn  # not needed anymore. Exact one is faster by now

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.return_self = return_self
        self.sumwnorm = sumwnorm
        self.feature_activation = feature_activation
        self.use_approximate_knn = use_approximate_knn
        self.use_dynamic_knn = use_dynamic_knn
        self.debug = debug
        self.n_knn_bins = n_knn_bins

        self.n_propagate = n_propagate
        self.n_prop_total = 2 * self.n_propagate

        with tf.name_scope(self.name + "/1/"):
            self.input_feature_transform = tf.keras.layers.Dense(
                n_propagate,
                activation=feature_activation,
                kernel_initializer="he_normal",
            )

        with tf.name_scope(self.name + "/2/"):
            s_kernel_initializer = "glorot_uniform"
            if coord_initialiser_noise is not None:
                s_kernel_initializer = EyeInitializer(
                    mean=0, stddev=coord_initialiser_noise
                )
            self.input_spatial_transform = tf.keras.layers.Dense(
                n_dimensions,
                # very slow turn on
                kernel_initializer=s_kernel_initializer,
                use_bias=False,
            )

        with tf.name_scope(self.name + "/3/"):
            self.output_feature_transform = tf.keras.layers.Dense(
                self.n_filters, activation="relu"
            )  # changed to relu

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        with tf.name_scope(self.name + "/1/"):
            self.input_feature_transform.build(input_shape)

        with tf.name_scope(self.name + "/2/"):
            if len(input_shapes) == 3:  # extra coords
                c_shape = [s for s in input_shape]
                c_shape[-1] += input_shapes[2][-1]
                self.input_spatial_transform.build(c_shape)
            else:
                self.input_spatial_transform.build(input_shape)

        with tf.name_scope(self.name + "/3/"):
            self.output_feature_transform.build(
                (input_shape[0], self.n_prop_total + input_shape[1])
            )

        with tf.name_scope(self.name + "/4/"):
            self.dynamic_radius = self.add_weight(
                name="dynamic_radius",
                initializer=tf.constant_initializer(1.0),
                trainable=False,
            )

        super(RaggedGravNet, self).build(input_shape)

    # @tf.function
    def create_output_features(self, x, neighbour_indices, distancesq):
        allfeat = []
        features = x

        features = self.input_feature_transform(features)
        prev_feat = features
        features = self.collect_neighbours(features, neighbour_indices, distancesq)
        features = tf.reshape(features, [-1, prev_feat.shape[1] * 2])
        features -= tf.tile(prev_feat, [1, 2])
        allfeat.append(features)

        features = tf.concat(allfeat + [x], axis=-1)
        return self.output_feature_transform(features)

    # @tf.function(reduce_retracing=True) #don't know why this is being retraced so often..
    def priv_call(self, x, row_splits, x_coord):

        coordinates = self.input_spatial_transform(x_coord)
        neighbour_indices, distancesq, sidx, sdist = (
            self.compute_neighbours_and_distancesq(coordinates, row_splits)
        )
        neighbour_indices = tf.reshape(
            neighbour_indices, [-1, self.n_neighbours]
        )  # for proper output shape for keras
        distancesq = tf.reshape(distancesq, [-1, self.n_neighbours])

        outfeats = self.create_output_features(x, neighbour_indices, distancesq)
        if self.return_self:
            neighbour_indices, distancesq = sidx, sdist
        return outfeats, coordinates, neighbour_indices, distancesq

    def call(self, inputs, training=None):
        x = inputs[0]
        row_splits = inputs[1]
        x_coord = x
        if len(inputs) == 3:
            x_coord = tf.concat([inputs[2], x], axis=-1)

        return self.priv_call(x, row_splits, x_coord)

    def compute_output_shape(self, input_shapes):
        if self.return_self:
            return (
                (input_shapes[0][0], 2 * self.n_filters),
                (input_shapes[0][0], self.n_dimensions),
                (input_shapes[0][0], self.n_neighbours + 1),
                (input_shapes[0][0], self.n_neighbours + 1),
            )
        else:
            return (
                (input_shapes[0][0], 2 * self.n_filters),
                (input_shapes[0][0], self.n_dimensions),
                (input_shapes[0][0], self.n_neighbours),
                (input_shapes[0][0], self.n_neighbours),
            )

    # @tf.function
    def compute_neighbours_and_distancesq(self, coordinates, row_splits):

        idx, dist = BinnedSelectKnn(
            self.n_neighbours + 1,
            coordinates,
            row_splits,
            max_radius=-1.0,
            tf_compatible=False,
            n_bins=self.n_knn_bins,
            name=self.name,
        )
        idx = tf.reshape(idx, [-1, self.n_neighbours + 1])
        dist = tf.reshape(dist, [-1, self.n_neighbours + 1])

        dist = tf.where(idx < 0, 0.0, dist)

        return idx[:, 1:], dist[:, 1:], idx, dist

    def collect_neighbours(self, features, neighbour_indices, distancesq):
        f = None
        if self.sumwnorm:
            f, _ = AccumulateKnnSumw(
                10.0 * distancesq, features, neighbour_indices, mean_and_max=True
            )
        else:
            f, _ = AccumulateKnn(
                10.0 * distancesq, features, neighbour_indices, mean_and_max=True
            )
        return f

    def get_config(self):
        config = {
            "n_neighbours": self.n_neighbours,
            "n_dimensions": self.n_dimensions,
            "n_filters": self.n_filters,
            "n_propagate": self.n_propagate,
            "return_self": self.return_self,
            "sumwnorm": self.sumwnorm,
            "feature_activation": self.feature_activation,
            "use_approximate_knn": self.use_approximate_knn,
            "n_knn_bins": self.n_knn_bins,
        }
        base_config = super(RaggedGravNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def set_all_gn_space_trainable(model, trainable=True):
        for l in model.layers:
            if isinstance(l, RaggedGravNet):
                l.input_spatial_transform.trainable = trainable


class TranslationInvariantMP(tf.keras.layers.Layer):
    def __init__(
        self,
        n_feature_transformation,
        activation="elu",
        mean=True,
        layer_norm=False,
        sum_weight=False,
        **kwargs,
    ):
        super(TranslationInvariantMP, self).__init__(**kwargs)

        self.n_feature_transformation = n_feature_transformation
        self.activation = activation
        initializer = "he_normal"
        self.mean = mean
        self.layer_norm = layer_norm
        self.sum_weight = sum_weight
        self.feature_tranformation_dense = []
        for i in range(len(self.n_feature_transformation)):
            with tf.name_scope(self.name + "/" + str(i)):
                self.feature_tranformation_dense.append(
                    tf.keras.layers.Dense(
                        n_feature_transformation[i],
                        activation=activation,
                        kernel_initializer=initializer,
                        trainable=self.trainable,
                        use_bias=i > 0,
                    )
                )

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        with tf.name_scope(self.name + "/" + str(0)):
            self.feature_tranformation_dense[0].build(input_shape)

        for i in range(1, len(self.feature_tranformation_dense)):
            with tf.name_scope(self.name + "/" + str(i)):
                self.feature_tranformation_dense[i].build(
                    (input_shape[0], self.n_feature_transformation[i - 1])
                )

        super(TranslationInvariantMP, self).build(input_shapes)

    def compute_output_shape(self, inputs_shapes):
        fshape = inputs_shapes[0][-1]
        return (None, sum(self.n_feature_transformation))

    def get_config(self):
        config = {
            "n_feature_transformation": self.n_feature_transformation,
            "activation": self.activation,
            "mean": self.mean,
            "layer_norm": self.layer_norm,
            "sum_weight": self.sum_weight,
        }
        base_config = super(TranslationInvariantMP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _trf_loop(self, features, neighbour_indices, distancesq, K, first):

        if self.layer_norm:
            features = layernorm(features)
        prev_feat = features
        if self.mean:
            # add a 1 to the features for translation invariance later
            if first:
                ones = tf.ones_like(features[:, 0:1])
                features = tf.concat([ones, features], axis=-1)
            # Standard Message Passing
            features = (
                AccumulateKnn(
                    10.0 * distancesq, features, neighbour_indices, mean_and_max=False
                )[0]
                * K
            )

            # this is only necessary for the first exchange, afterwards the features are already translation independent
            if first:
                minus_xi = features[:, 0:1]
                features = features[:, 1:]
                features -= prev_feat * minus_xi
            if self.sum_weight:
                wsum = tf.math.divide_no_nan(
                    K,
                    tf.reduce_sum(tf.exp(-10.0 * distancesq), axis=1, keepdims=True)
                    + 1e-2,
                )  # large eps
                features *= wsum
        else:  # max
            nfeat = SelectWithDefault(neighbour_indices, features, -2.0)
            features = (
                tf.reduce_max(
                    tf.exp(-10.0 * distancesq) * nfeat - features[:, tf.newaxis, :],
                    axis=1,
                )
                * K
            )

        return features / K

    def create_output_features(self, x, neighbour_indices, distancesq):
        allfeat = []
        features = x
        K = tf.cast(tf.shape(neighbour_indices)[1], "float32")

        for i in range(len(self.n_feature_transformation)):
            features = self._trf_loop(
                features, neighbour_indices, distancesq, K, i == 0
            )
            t = self.feature_tranformation_dense[i]
            features = t(features)  # divide by K here again
            allfeat.append(features)

        features = tf.concat(allfeat, axis=-1)
        features = tf.reshape(features, [-1, sum(self.n_feature_transformation)])
        return features

    # @tf.function
    def call(self, inputs):
        if len(inputs) == 3:
            x, neighbor_indices, distancesq = inputs
        elif len(inputs) == 2:
            x, neighbor_indices = inputs
            distancesq = tf.zeros_like(neighbor_indices, dtype="float32")
        else:
            raise ValueError(self.name + " was passed wrong inputs")

        return self.create_output_features(x, neighbor_indices, distancesq)


class DistanceWeightedMessagePassing(tf.keras.layers.Layer):
    """
    Inputs: x, neighbor_indices, distancesq

    """

    def __init__(
        self,
        n_feature_transformation,  # =[32, 32, 32, 32, 4, 4],
        sumwnorm=False,
        activation="relu",
        exp_distances=True,  # use feat * exp(-distance) weighting, if not simple feat * distance
        **kwargs,
    ):
        super(DistanceWeightedMessagePassing, self).__init__(**kwargs)

        self.n_feature_transformation = n_feature_transformation
        self.sumwnorm = sumwnorm
        self.feature_tranformation_dense = []
        self.activation = activation
        self.exp_distances = exp_distances
        for i in range(len(self.n_feature_transformation)):
            with tf.name_scope(self.name + "/5/" + str(i)):
                self.feature_tranformation_dense.append(
                    tf.keras.layers.Dense(
                        self.n_feature_transformation[i], activation=activation
                    )
                )  # restrict variations a bit

    def build(self, input_shapes):
        input_shape = input_shapes[0]

        with tf.name_scope(self.name + "/5/" + str(0)):
            self.feature_tranformation_dense[0].build(input_shape)

        for i in range(1, len(self.feature_tranformation_dense)):
            with tf.name_scope(self.name + "/5/" + str(i)):
                self.feature_tranformation_dense[i].build(
                    (input_shape[0], self.n_feature_transformation[i - 1] * 2)
                )

        super(DistanceWeightedMessagePassing, self).build(input_shapes)

    def compute_output_shape(self, inputs_shapes):
        fshape = inputs_shapes[0][-1]
        return (None, fshape + 2 * sum(self.n_feature_transformation))

    def get_config(self):
        config = {
            "n_feature_transformation": self.n_feature_transformation,
            "activation": self.activation,
            "exp_distances": self.exp_distances,
            "sumwnorm": self.sumwnorm,
        }
        base_config = super(DistanceWeightedMessagePassing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # @tf.function
    def create_output_features(self, x, neighbour_indices, distancesq):
        allfeat = []
        features = x

        for i in range(len(self.n_feature_transformation)):
            t = self.feature_tranformation_dense[i]
            features = t(features)
            prev_feat = features
            features = self.collect_neighbours(features, neighbour_indices, distancesq)
            features = tf.reshape(features, [-1, prev_feat.shape[1] * 2])
            features -= tf.tile(prev_feat, [1, 2])
            allfeat.append(features)

        features = tf.concat(allfeat + [x], axis=-1)
        return features

    def collect_neighbours(self, features, neighbour_indices, distancesq):
        f = None
        if self.sumwnorm:
            if self.exp_distances:
                f, _ = AccumulateKnnSumw(
                    10.0 * distancesq, features, neighbour_indices, mean_and_max=True
                )
            else:
                f, _ = AccumulateLinKnnSumw(
                    distancesq, features, neighbour_indices, mean_and_max=True
                )
        else:
            if self.exp_distances:
                f, _ = AccumulateKnn(10.0 * distancesq, features, neighbour_indices)
            else:
                f, _ = AccumulateLinKnn(distancesq, features, neighbour_indices)
        return f

    def call(self, inputs):
        x, neighbor_indices, distancesq = inputs
        return self.create_output_features(x, neighbor_indices, distancesq)


class XYZtoXYZPrime(tf.keras.layers.Layer):

    def __init__(self, new_prime, **kwargs):
        super(XYZtoXYZPrime, self).__init__(**kwargs)
        self.new_prime = new_prime

    def get_config(self):
        config = {"new_prime": self.new_prime}
        base_config = super(XYZtoXYZPrime, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        x = inputs[..., 0:1]
        y = inputs[..., 1:2]
        z = inputs[..., 2:3]
        r = tf.sqrt(tf.reduce_sum(inputs**2, axis=-1, keepdims=True) + 1e-2)

        # also adjust scale a bit
        xprime = x / tf.where(z == 0.0, tf.sign(z) * 1.0, z * 10.0)
        yprime = y / tf.where(z == 0.0, tf.sign(z) * 1.0, z * 10.0)
        zprime = r / 100.0
        if self.new_prime:
            xprime = xprime * 10.0
            yprime = yprime * 10.0

        return tf.concat([xprime, yprime, zprime], axis=-1)


class RandomSampling(tf.keras.layers.Layer):
    """
    Random sampling layer for ragged tensors

    :param reduction: float, reduction factor
    :param epsilon: float, small number to counter machine precision errors
    :param **kwargs: other arguments

    :return: selected points, row_splits, [old_size, indices_selected]

    Layer that randomly samples points from a ragged point cloud.
    On average it reduces the number of points by `reduction` factor.
    If the reduction factor is chosen to be very high it is possible to
    lose all points in a single event. In this case the reduction factor
    will be changed on the fly to the hightes value that still leaves at
    least one event in every point cloud.
    """

    def __init__(self, reduction=10.0, epsilon=1e-7, **kwargs):
        super(RandomSampling, self).__init__(**kwargs)
        self.reduction = reduction
        self.epsilon = epsilon

    def get_config(self):
        config = super(RandomSampling, self).get_config()
        config.update({"reduction": self.reduction, "epsilon": self.epsilon})
        return config

    def call(self, inputs):
        """
        We keep score <= 1/reduction points and discard the rest.
        """
        x, is_track, row_splits = inputs

        assert len(row_splits.shape) == 1, "row_splits must be 1D"
        assert len(is_track.shape) == 2, "is_track must be 2D"
        assert len(x.shape) == 2, "x must be 2D"
        assert is_track.shape[0] == x.shape[0], "x and is_track don't match"
        # assert row_splits[-1] == x.shape[0], "row_splits and x don't match"
        is_track = tf.cast(is_track, tf.bool)

        N = row_splits[-1]
        ragged = tf.RaggedTensor.from_row_splits(x, row_splits)

        score = tf.random.uniform(shape=(N, 1), minval=0, maxval=1)
        score = tf.where(is_track, 0.0, score)
        score_ragged = tf.RaggedTensor.from_row_splits(score, row_splits)

        min_event_score = tf.reduce_min(score_ragged, axis=1)  # events x 1
        highest_min_event_score = tf.reduce_max(min_event_score)
        threshold = tf.where(
            highest_min_event_score < 1.0 / self.reduction,
            1.0 / self.reduction,
            highest_min_event_score + self.epsilon,
        )
        if threshold != 1.0 / self.reduction:
            print("WARNING: reduction threshold had to be adapted")

        mask = score_ragged <= threshold
        mask = tf.squeeze(score_ragged <= threshold, axis=-1)
        selected = tf.ragged.boolean_mask(ragged, mask)
        new_rs = selected.row_splits
        n_batch = tf.shape(score)[0]
        full_indices = tf.range(n_batch, dtype=tf.int32)[:, tf.newaxis]
        indices_selected = full_indices[score <= threshold]
        old_size = row_splits[-1]
        old_size = tf.cast(old_size, tf.int32)
        indices_selected = tf.cast(indices_selected, tf.int32)

        x = tf.gather(x, indices_selected, axis=0)

        return x, new_rs, [old_size, indices_selected[..., tf.newaxis]]
