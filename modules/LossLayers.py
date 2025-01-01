import time
import tensorflow as tf
from object_condensation import OC_loss
from oc_helper_ops import SelectWithDefault
from oc_helper_ops import CreateMidx
from baseModules import LayerWithMetrics
from ragged_tools import normalise_index


def smooth_max(var, eps=1e-3, **kwargs):
    return eps * tf.reduce_logsumexp(var / eps, **kwargs)


def one_hot_encode_id(t_pid, n_classes):
    valued_pids = tf.zeros_like(t_pid) + 4  # defaults to 4 as unkown
    valued_pids = tf.where(
        tf.math.logical_or(t_pid == 22, tf.abs(t_pid) == 11), 0, valued_pids
    )  # isEM

    valued_pids = tf.where(tf.abs(t_pid) == 211, 1, valued_pids)  # isHad
    valued_pids = tf.where(tf.abs(t_pid) == 2212, 1, valued_pids)  # proton isChHad
    valued_pids = tf.where(tf.abs(t_pid) == 321, 1, valued_pids)  # K+

    valued_pids = tf.where(tf.abs(t_pid) == 13, 2, valued_pids)  # isMIP

    valued_pids = tf.where(
        tf.abs(t_pid) == 111, 3, valued_pids
    )  # pi0 isNeutrHadOrOther
    valued_pids = tf.where(
        tf.abs(t_pid) == 2112, 3, valued_pids
    )  # neutron isNeutrHadOrOther
    valued_pids = tf.where(tf.abs(t_pid) == 130, 3, valued_pids)  # K0 isNeutrHadOrOther
    valued_pids = tf.where(tf.abs(t_pid) == 310, 3, valued_pids)  # K0 short
    valued_pids = tf.where(
        tf.abs(t_pid) == 3122, 3, valued_pids
    )  # lambda isNeutrHadOrOther

    valued_pids = tf.cast(valued_pids, tf.int32)[:, 0]

    known = tf.where(valued_pids == 4, tf.zeros_like(valued_pids), 1)
    valued_pids = tf.where(known < 1, 3, valued_pids)  # set to 3
    known = tf.expand_dims(known, axis=1)  # V x 1 style

    depth = (
        n_classes  # If n_classes=pred_id.shape[1], should we add an assert statement?
    )
    return tf.one_hot(valued_pids, depth), known


def huber(x, d):
    losssq = x**2
    absx = tf.abs(x)
    losslin = d**2 + 2.0 * d * (absx - d)
    return tf.where(absx < d, losssq, losslin)


def quantile(x, tau):
    return tf.maximum(tau * x, (tau - 1) * x)


def _calc_energy_weights(t_energy, t_pid=None, upmouns=True, alt_energy_weight=True):

    lower_cut = 0.5
    w = tf.where(
        t_energy > 10.0,
        1.0,
        ((t_energy - lower_cut) / 10.0) * 10.0 / (10.0 - lower_cut),
    )
    w = tf.nn.relu(w)
    if alt_energy_weight:
        extra = t_energy / 50.0 + 0.01
        # extra = tf.math.divide_no_nan(extra, tf.reduce_sum(extra,axis=0,keepdims=True)+1e-3)
        w *= extra
    return w


def _calc_energy_weights_quadratic(
    t_energy, t_pid=None, upmouns=True, alt_energy_weight=True
):

    w = (t_energy / 33.0) ** 2  # 33 is an empirical factor for 200 pile-up events
    return w


class NormaliseTruthIdxs(tf.keras.layers.Layer):

    def __init__(self, active=True, add_rs_offset=True, **kwargs):
        """
        changes arbitrary truth indices to well defined indices such that
        sort(unique(t_idx)) = -1, 0, 1, 2, 3, 4, 5, ... for each row split

        This should be called after every layer that could have modified
        the truth indices or removed hits, if the output needs to be regular.

        This Layer takes < 10ms usually so can be used generously.

        :param active: determines if it should be active.
                       In pure inference mode that might not be needed

        Inputs: truth indices, row splits
        Output: new truth indices

        """
        if "dynamic" in kwargs:
            super(NormaliseTruthIdxs, self).__init__(**kwargs)
        else:
            super(NormaliseTruthIdxs, self).__init__(dynamic=True, **kwargs)

        self.active = active
        self.add_rs_offset = add_rs_offset

    def get_config(self):
        config = {"active": self.active, "add_rs_offset": self.add_rs_offset}
        base_config = super(NormaliseTruthIdxs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def call(self, inputs):
        assert len(inputs) == 2
        t_idx, rs = inputs

        # double unique
        if not self.active or rs.shape[0] == None:
            return t_idx

        return normalise_index(t_idx, rs, self.add_rs_offset)


class LossLayerBase(LayerWithMetrics):
    """Base class for HGCalML loss layers.

    Use the 'active' switch to switch off the loss calculation.
    This needs to be done by hand, and is not handled by the TF 'training' flag, since
    it might be desirable to
     (a) switch it off during training, or
     (b) calculate the loss also during inference (e.g. validation)


    The 'scale' argument determines a global sale factor for the loss.
    """

    def __init__(
        self,
        active=True,
        scale=1.0,
        print_loss=False,
        print_batch_time=False,
        return_lossval=False,
        print_time=False,  # compat, has no effect
        record_batch_time=False,
        **kwargs
    ):
        super(LossLayerBase, self).__init__(**kwargs)

        if print_time:
            print("print_time has no effect and is only for compatibility purposes")

        self.active = active
        self.scale = scale
        self.print_loss = print_loss
        self.print_batch_time = print_batch_time
        self.record_batch_time = record_batch_time
        self.return_lossval = return_lossval
        with tf.init_scope():
            now = tf.timestamp()
        self.time = tf.Variable(-now, name=self.name + "_time", trainable=False)

    def get_config(self):
        config = {
            "active": self.active,
            "scale": self.scale,
            "print_loss": self.print_loss,
            "print_batch_time": self.print_batch_time,
            "record_batch_time": self.record_batch_time,
            "return_lossval": self.return_lossval,
        }
        base_config = super(LossLayerBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(LossLayerBase, self).build(input_shape)

    def create_safe_zero_loss(self, a):
        zero_loss = tf.zeros_like(tf.reduce_mean(a))
        zero_loss = tf.where(tf.math.is_finite(zero_loss), zero_loss, 0.0)
        return zero_loss

    def call(self, inputs):
        lossval = tf.constant([0.0], dtype="float32")
        a = None
        if not isinstance(inputs, list):
            a = inputs
        elif isinstance(inputs, list) and len(inputs) == 2:
            a, _ = inputs
        elif isinstance(inputs, list) and len(inputs) > 2:
            a, *_ = inputs
        else:
            raise ValueError("LossLayerBase: input not understood")

        if self.active:
            now = tf.timestamp()

            # check for empty batches

            # generally protect against empty batches
            if a.shape[0] is None or a.shape[0] == 0:
                zero_loss = self.create_safe_zero_loss(a)
                print(self.name, "returning zero loss", zero_loss)
                lossval = zero_loss
            else:
                lossval = self.scale * self.loss(inputs)

            self.maybe_print_loss(lossval, now)

            # lossval = tf.debugging.check_numerics(lossval, self.name+" produced inf or nan.")
            # this can happen for empty batches. If there are deeper problems, check in the losses themselves
            # lossval = tf.where(tf.math.is_finite(lossval), lossval ,0.)
            if not self.return_lossval:
                self.add_loss(lossval)

        self.wandb_log({self.name + "_loss": self._to_numpy(lossval)})
        if self.return_lossval:
            return a, lossval
        else:
            return a

    def _to_numpy(self, tensor):
        # Converts a tensor to numpy regardless of execution mode
        if tf.executing_eagerly():
            return tensor.numpy()
        else:
            with tf.compat.v1.Session() as sess:
                return sess.run(tensor)

    def wandb_log(self, log_dict):  # restrict to dict, translate all to numpy if needed
        for key, value in log_dict.items():
            # check if it is a tensor
            if isinstance(value, tf.Tensor):
                log_dict[key] = self._to_numpy(value)
        # ru super wandb_log
        super(LossLayerBase, self).wandb_log(log_dict)

    def loss(self, inputs):
        """
        Overwrite this function in derived classes.
        Input: always a list of inputs, the first entry in the list will be returned, and should be the features.
        The rest is free (but will probably contain the truth somewhere)
        """
        return tf.constant(0.0, dtype="float32")

    def maybe_print_loss(self, lossval, stime=None):
        if self.print_loss:
            if hasattr(lossval, "numpy"):
                print(self.name, "loss", lossval.numpy())
                tf.print(self.name, "loss", lossval.numpy())
            else:
                tf.print(self.name, "loss", lossval)
                print(self.name, "loss", lossval)

        if self.print_batch_time or self.record_metrics:
            now = tf.timestamp()
            prev = self.time
            prev = tf.where(prev < 0.0, now, prev)
            batchtime = now - prev  # round((time.time()-self.time)*1000.)/1000.
            losstime = 0.0
            if stime is not None:
                losstime = now - stime
            tf.keras.backend.update(self.time, now)
            if self.print_batch_time:
                tf.print(self.name, "batch time", batchtime * 1000.0, "ms")
                if stime is not None:
                    tf.print(self.name, "loss time", losstime * 1000.0, "ms")
                print(self.name, "batch time", batchtime * 1000.0, "ms")
                if stime is not None:
                    print(self.name, "loss time", losstime * 1000.0, "ms")
            if self.record_batch_time and self.record_metrics:
                self.add_prompt_metric(batchtime, self.name + "_batch_time")
                if stime is not None:
                    self.add_prompt_metric(losstime, self.name + "_loss_time")

    def compute_output_shape(self, input_shapes):
        if self.return_lossval:
            return input_shapes[0], (None,)
        else:
            return input_shapes[0]


class LLDummy(LossLayerBase):

    def loss(self, inputs):
        return tf.reduce_mean(inputs**2)


class LLValuePenalty(LossLayerBase):

    def __init__(self, default: float = 0.0, **kwargs):
        """
        Simple value penalty loss, tries to keep values around default using simple
        L2 regularisation; returns input
        """

        super(LLValuePenalty, self).__init__(**kwargs)
        self.default = default

    def get_config(self):
        config = {"default": self.default}
        base_config = super(LLValuePenalty, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def loss(self, inputs):
        lossval = tf.reduce_mean((self.default - inputs) ** 2)
        return tf.where(tf.math.is_finite(lossval), lossval, 0.0)  # DEBUG


class CreateTruthSpectatorWeights(tf.keras.layers.Layer):
    def __init__(self, threshold, minimum, active, **kwargs):
        """
        active: does not enable a loss, but acts similar to other layers using truth information
                      as a switch to not require truth information at all anymore (for inference)

        Inputs: spectator score, truth indices
        Outputs: spectator weights (1-minimum above threshold, 0 else)

        """
        super(CreateTruthSpectatorWeights, self).__init__(**kwargs)
        self.threshold = threshold
        self.minimum = minimum
        self.active = active

    def get_config(self):
        config = {
            "threshold": self.threshold,
            "minimum": self.minimum,
            "active": self.active,
        }
        base_config = super(CreateTruthSpectatorWeights, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def call(self, inputs):
        if not self.active:
            return inputs[0]

        abovethresh = inputs[0] > self.threshold
        # notnoise = inputs[1] >= 0
        # noise can never be spectator
        return tf.where(abovethresh, tf.ones_like(inputs[0]) - self.minimum, 0.0)


class LLRegulariseGravNetSpace(LossLayerBase):

    def __init__(self, project=False, **kwargs):
        """
        Regularisation layer (not truth dependent)
        Regularises the GravNet space to have similar distances than the physical space

        Inputs:
        - GravNet Distances
        - physical coordinates (prime_coords)
        - neighbour indices

        Outputs:
        - GravNet Distances (unchanged)

        Options:
        - project: projects the physical inputs to a one-sphere (not useful when using HGCAL 'prime coordinates')

        """
        super(LLRegulariseGravNetSpace, self).__init__(**kwargs)
        self.project = project
        print(
            "INFO: LLRegulariseGravNetSpace: this is actually a regulariser: move to right file soon."
        )

    def get_config(self):
        config = {"project": self.project}
        base_config = super(LLRegulariseGravNetSpace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def loss(self, inputs):
        assert len(inputs) == 3
        gndist, in_coords, nidx = inputs

        if self.project:
            in_coords = in_coords / tf.sqrt(
                tf.reduce_sum(in_coords**2, axis=1, keepdims=True) + 1e-6
            )
        else:
            # this is prime-coords, so x', y', z', where z is projected towards the beamspot
            # so we can just remove it
            in_coords = in_coords[:, :2]

        ncoords = SelectWithDefault(nidx, in_coords, 0.0)
        dist = tf.reduce_sum(
            (in_coords[:, tf.newaxis, :] - ncoords) ** 2, axis=2
        )  # V x K+1

        dist = tf.where(nidx < 1, 0.0, dist)  # mask masked again
        gndist = tf.where(
            nidx < 1, 0.0, gndist
        )  # mask masked again (this should already be done but just in case)

        # old goes here
        dist = tf.sqrt(dist + 1e-6)
        gndist = tf.sqrt(gndist + 1e-6)

        # now just normalise with softmax
        dist = tf.nn.softmax(dist, axis=1)
        gndist = tf.nn.softmax(gndist, axis=1)

        lossval = tf.reduce_mean((dist - gndist) ** 2)  # both should be similar

        return lossval


class LLFillSpace(LossLayerBase):
    def __init__(self, maxhits: int = 1000, runevery: int = -1, **kwargs):
        """
        calculated a PCA of all points in coordinate space and
        penalises very asymmetric PCs.
        Reduces the risk of falling back to a (hyper)surface

        Inputs:
         - coordinates, row splits, (truth index - optional. then only applied to non-noise)
        Outputs:
         - coordinates (unchanged)
        """
        print(
            "INFO: LLFillSpace: this is actually a regulariser: move to right file soon."
        )
        assert maxhits > 0
        self.maxhits = maxhits
        self.runevery = runevery
        self.counter = -1
        if runevery < 0:
            self.counter = -2
        if "dynamic" in kwargs:
            super(LLFillSpace, self).__init__(**kwargs)
        else:
            super(LLFillSpace, self).__init__(dynamic=True, **kwargs)

    def get_config(self):
        config = {"maxhits": self.maxhits, "runevery": self.runevery}
        base_config = super(LLFillSpace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def _rs_loop(coords, tidx, maxhits=1000):
        # only select a few hits to keep memory managable
        nhits = coords.shape[0]
        sel = None
        if nhits > maxhits:
            sel = tf.random.uniform(
                shape=(maxhits,), minval=0, maxval=coords.shape[0] - 1, dtype=tf.int32
            )
        else:
            sel = tf.range(coords.shape[0], dtype=tf.int32)
        sel = tf.expand_dims(sel, axis=1)
        coords = tf.gather_nd(coords, sel)  # V' x C
        if tidx is not None:
            tidx = tf.gather_nd(tidx, sel)  # V' x C
            coords = coords[tidx[:, 0] >= 0]
        # print('coords',coords.shape)
        means = tf.reduce_mean(coords, axis=0, keepdims=True)  # 1 x C
        coords -= means  # V' x C
        # build covariance
        cov = tf.expand_dims(coords, axis=1) * tf.expand_dims(
            coords, axis=2
        )  # V' x C x C
        cov = tf.reduce_mean(cov, axis=0, keepdims=False)  # 1 x C x C
        # print('cov',cov)
        # get eigenvals
        eigenvals, _ = tf.linalg.eig(cov)  # cheap because just once, no need for approx
        eigenvals = tf.cast(eigenvals, dtype="float32")
        # penalise one small EV (e.g. when building a surface)
        pen = tf.math.log(
            (
                tf.math.divide_no_nan(
                    tf.reduce_mean(eigenvals), tf.reduce_min(eigenvals) + 1e-6
                )
                - 1.0
            )
            ** 2
            + 1.0
        )
        return pen

    @staticmethod
    def raw_loss(coords, rs, tidx, maxhits=1000):
        loss = tf.zeros([], dtype="float32")
        for i in range(len(rs) - 1):
            rscoords = coords[rs[i] : rs[i + 1]]
            loss += LLFillSpace._rs_loop(rscoords, tidx, maxhits)
        return tf.math.divide_no_nan(loss, tf.cast(rs.shape[0], dtype="float32"))

    def loss(self, inputs):
        assert len(inputs) == 2 or len(inputs) == 3  # coords, rs
        tidx = None
        if len(inputs) == 3:
            coords, rs, tidx = inputs
        else:
            coords, rs = inputs
        if self.counter >= 0:  # completely optimise away increment
            if self.counter < self.runevery:
                self.counter += 1
                return tf.zeros_like(coords[0, 0])
            self.counter = 0
        lossval = LLFillSpace.raw_loss(coords, rs, tidx, self.maxhits)

        if self.counter == -1:
            self.counter += 1
        return lossval


class LLClusterCoordinates(LossLayerBase):
    """
    Cluster using truth index and coordinates
    Inputs:
    - coordinates
    - truth index
    - row splits
    """

    def __init__(
        self,
        downsample: int = -1,
        ignore_noise=False,
        hinge_mode=False,
        specweight_to_weight=False,
        **kwargs
    ):
        if "dynamic" in kwargs:
            super(LLClusterCoordinates, self).__init__(**kwargs)
        else:
            super(LLClusterCoordinates, self).__init__(dynamic=True, **kwargs)

        self.downsample = downsample
        self.ignore_noise = ignore_noise
        self.hinge_mode = hinge_mode
        self.specweight_to_weight = specweight_to_weight

        # self.built = True #not necessary for loss layers

    def get_config(self):
        base_config = super(LLClusterCoordinates, self).get_config()
        return dict(
            list(base_config.items())
            + list(
                {
                    "downsample": self.downsample,
                    "ignore_noise": self.ignore_noise,
                    "hinge_mode": self.hinge_mode,
                    "specweight_to_weight": self.specweight_to_weight,
                }.items()
            )
        )

    def _attfunc(self, dsq):
        if self.hinge_mode:
            return tf.sqrt(dsq + 1e-4)
        return tf.math.log(tf.math.exp(1.0) * dsq + 1.0)

    def _rep_func(self, dsq):
        if self.hinge_mode:
            return tf.nn.relu(1.0 - tf.sqrt(dsq + 1e-6))

        return tf.exp(-tf.sqrt(dsq + 1e-4) / (5.0))

    # all this needs some cleaning up
    def _rs_loop(self, coords, tidx, specweight, energy):

        if tidx.shape[0] == 0:
            print(self.name, "batch empty")
            return 3 * [self.create_safe_zero_loss(coords)]

        Msel, M_not, N_per_obj = CreateMidx(tidx, calc_m_not=True)  # N_per_obj: K x 1

        if Msel is None or N_per_obj is None:
            print(self.name, "no objects in batch")
            return 3 * [self.create_safe_zero_loss(coords)]

        # almost empty
        if Msel.shape[0] == 0 or (Msel.shape[0] == 1 and Msel.shape[1] == 1):
            print(self.name, "just one point left")
            return 3 * [self.create_safe_zero_loss(coords)]

        if self.ignore_noise:
            e_tidx = tf.cast(tf.expand_dims(tidx, axis=0), "float32")
            e_tidx *= M_not
            M_not = tf.where(e_tidx < 0, 0.0, M_not)

        N_per_obj = tf.cast(N_per_obj, dtype="float32")
        N_tot = tf.cast(tidx.shape[0], dtype="float32")
        K = tf.cast(Msel.shape[0], dtype="float32")

        padmask_m = SelectWithDefault(
            Msel, tf.ones_like(coords[:, 0:1]), 0.0
        )  # K x V' x 1
        coords_m = SelectWithDefault(Msel, coords, 0.0)  # K x V' x C

        q = 1.0 - specweight  # *(1.+tf.math.log(tf.abs(energy)+1.))+1e-2
        q_m = SelectWithDefault(Msel, q, 0.0)  # K x V' x C
        q_k = tf.reduce_sum(q_m, axis=1)  # K x 1

        # create average
        av_coords_m = tf.reduce_sum(coords_m * padmask_m * q_m, axis=1)  # K x C
        av_coords_m = tf.math.divide_no_nan(
            av_coords_m, tf.reduce_sum(padmask_m * q_m, axis=1) + 1e-3
        )  # K x C
        av_coords_m = tf.expand_dims(av_coords_m, axis=1)  ##K x 1 x C

        distloss = tf.reduce_sum((av_coords_m - coords_m) ** 2, axis=2)
        distloss = q_m[:, :, 0] * self._attfunc(distloss) * padmask_m[:, :, 0]
        distloss = tf.math.divide_no_nan(
            q_k[:, 0] * tf.reduce_sum(distloss, axis=1), N_per_obj[:, 0] + 1e-3
        )  # K
        distloss = tf.math.divide_no_nan(
            tf.reduce_sum(distloss), tf.reduce_sum(q_k) + 1e-3
        )

        # check if Mnot is empty
        if M_not.shape[0] == 0 or tf.reduce_sum(M_not) == 0.0:
            print(self.name, "no repulsive loss")
            return distloss, distloss, self.create_safe_zero_loss(coords)

        repdist = tf.expand_dims(coords, axis=0) - av_coords_m  # K x V x C
        repdist = tf.expand_dims(q, axis=0) * tf.reduce_sum(
            repdist**2, axis=-1, keepdims=True
        )  # K x V x 1

        # add a long range part to it
        reploss = M_not * self._rep_func(repdist)  # K x V x 1
        # downweight noise
        reploss = (
            q_k * tf.reduce_sum(reploss, axis=1) / (N_tot - N_per_obj + 1e-3)
        )  # K x 1
        reploss = tf.reduce_sum(reploss) / (tf.reduce_sum(q_k) + 1e-3)

        return distloss + reploss, distloss, reploss

    def raw_loss(self, acoords, atidx, aspecw, aenergy, rs):

        lossval = tf.zeros_like(acoords[0, 0])
        reploss = tf.zeros_like(acoords[0, 0])
        distloss = tf.zeros_like(acoords[0, 0])

        if rs.shape[0] is None:
            return lossval, distloss, reploss

        nbatches = rs.shape[0] - 1
        for i in range(nbatches):
            coords = acoords[rs[i] : rs[i + 1]]
            tidx = atidx[rs[i] : rs[i + 1]]
            specw = aspecw[rs[i] : rs[i + 1]]
            energy = aenergy[rs[i] : rs[i + 1]]

            if self.downsample > 0:  # and self.downsample < coords.shape[0]:
                sel = tf.range(coords.shape[0])
                sel = tf.random.shuffle(sel)

                length = tf.reduce_min(
                    [tf.constant(self.downsample), tf.shape(coords)[0]]
                )

                sel = sel[:length]
                # sel = tf.random.uniform(shape=(self.downsample,), minval=0, maxval=coords.shape[0]-1, dtype=tf.int32)
                sel = tf.expand_dims(sel, axis=1)
                coords = tf.gather_nd(coords, sel)
                tidx = tf.gather_nd(tidx, sel)
                specw = tf.gather_nd(specw, sel)
                energy = tf.gather_nd(energy, sel)

            if self.specweight_to_weight:
                specw = 1.0 - specw

            tlv, tdl, trl = self._rs_loop(coords, tidx, specw, energy)
            tlv = tf.where(tf.math.is_finite(tlv), tlv, 0.0)
            tdl = tf.where(tf.math.is_finite(tdl), tdl, 0.0)
            trl = tf.where(tf.math.is_finite(trl), trl, 0.0)
            lossval += tlv
            distloss += tdl
            reploss += trl
        nbatches = tf.cast(nbatches, dtype="float32") + 1e-3
        return lossval / nbatches, distloss / nbatches, reploss / nbatches

    def loss(self, inputs):
        if len(inputs) == 5:
            coords, tidx, specw, energy, rs = inputs
        elif len(inputs) == 4:
            coords, tidx, energy, rs = inputs
            specw = tf.zeros_like(energy)
            if self.specweight_to_weight:
                specw = 1.0 - specw
        else:
            raise ValueError("LLClusterCoordinates: expects 4 or 5 inputs")

        lossval, distloss, reploss = self.raw_loss(coords, tidx, specw, energy, rs)

        lossval = tf.where(tf.math.is_finite(lossval), lossval, 0.0)  # DEBUG

        self.wandb_log(
            {
                self.name + "_att_loss": self.scale * distloss,
                self.name + "_rep_loss": self.scale * reploss,
            }
        )

        return lossval


class LLFullObjectCondensation(LossLayerBase):
    """
    Cluster using truth index and coordinates

    This is a copy of the above, reducing the nested function calls.

    keep the individual loss definitions as separate functions, even if they are trivial.
    inherit from this class to implement different variants of the loss ingredients without
    making the config explode (more)
    """

    def __init__(
        self,
        *,
        energy_loss_weight=1.0,
        use_energy_weights=True,
        alt_energy_weight=False,
        train_energy_correction=True,
        q_min=0.1,
        no_beta_norm=False,
        noise_q_min=None,
        potential_scaling=1.0,
        repulsion_scaling=1.0,
        s_b=1.0,
        position_loss_weight=1.0,
        classification_loss_weight=1.0,
        timing_loss_weight=1.0,
        use_spectators=True,
        beta_loss_scale=1.0,
        use_average_cc_pos=0.0,
        payload_rel_threshold=0.1,
        rel_energy_mse=False,
        smooth_rep_loss=False,
        pre_train=False,
        huber_energy_scale=-1.0,
        downweight_low_energy=True,
        n_ccoords=2,
        energy_den_offset=1.0,
        noise_scaler=1.0,
        too_much_beta_scale=0.0,
        cont_beta_loss=False,
        log_energy=False,
        n_classes=0,
        prob_repulsion=False,
        phase_transition=0.0,
        phase_transition_double_weight=False,
        alt_potential_norm=True,
        payload_beta_gradient_damping_strength=0.0,
        payload_beta_clip=0.0,
        kalpha_damping_strength=0.0,
        cc_damping_strength=0.0,
        standard_configuration=None,
        beta_gradient_damping=0.0,
        alt_energy_loss=False,
        repulsion_q_min=-1.0,
        super_repulsion=False,
        use_local_distances=True,
        energy_weighted_qmin=False,
        super_attraction=False,
        div_repulsion=False,
        dynamic_payload_scaling_onset=-0.005,
        beta_push=0.0,
        implementation="",
        global_weight=False,
        **kwargs
    ):
        """
        Read carefully before changing parameters

        :param energy_loss_weight:
        :param use_energy_weights:
        :param q_min:
        :param no_beta_norm:
        :param potential_scaling:
        :param repulsion_scaling:
        :param s_b:
        :param position_loss_weight:
        :param classification_loss_weight:
        :param timing_loss_weight:
        :param use_spectators:
        :param beta_loss_scale:
        :param use_average_cc_pos: weight (between 0 and 1) of the average position vs. the kalpha position
        :param payload_rel_threshold:
        :param rel_energy_mse:
        :param smooth_rep_loss:
        :param pre_train:
        :param huber_energy_scale:
        :param downweight_low_energy:
        :param n_ccoords:
        :param energy_den_offset:
        :param noise_scaler:
        :param too_much_beta_scale:
        :param cont_beta_loss:
        :param log_energy:
        :param n_classes: give the real number of classes, in the truth labelling, class 0 is always ignored so if you
                          have 6 classes, label them from 1 to 6 not 0 to 5. If n_classes is 0, no classification loss
                          is applied
        :param prob_repulsion
        :param phase_transition
        :param standard_configuration:
        :param alt_energy_loss: introduces energy loss with very mild gradient for large delta. (modified 1-exp form)
        :param dynamic_payload_scaling_onset: only apply payload loss to well reconstructed showers. typical values 0.1 (negative=off)
        :param kwargs:
        """
        if "dynamic" in kwargs:
            super(LLFullObjectCondensation, self).__init__(**kwargs)
        else:
            super(LLFullObjectCondensation, self).__init__(dynamic=True, **kwargs)

        assert use_local_distances  # fixed now, if they should not be used, pass 1s

        if too_much_beta_scale == 0 and cont_beta_loss:
            raise ValueError("cont_beta_loss must be used with too_much_beta_scale>0")

        if huber_energy_scale > 0 and alt_energy_loss:
            raise ValueError(
                "huber_energy_scale>0 and alt_energy_loss exclude each other"
            )

        from object_condensation import (
            Basic_OC_per_sample,
            PushPull_OC_per_sample,
            PreCond_OC_per_sample,
        )
        from object_condensation import (
            Hinge_OC_per_sample_damped,
            Hinge_OC_per_sample,
            Hinge_Manhatten_OC_per_sample,
        )
        from object_condensation import (
            Dead_Zone_Hinge_OC_per_sample,
            Hinge_OC_per_sample_learnable_qmin,
            Hinge_OC_per_sample_learnable_qmin_betascale_position,
        )

        impl = Basic_OC_per_sample
        if implementation == "pushpull":
            impl = PushPull_OC_per_sample
        if implementation == "precond":
            impl = PreCond_OC_per_sample
        if implementation == "hinge":
            impl = Hinge_OC_per_sample
        if implementation == "hinge_deadzone":
            impl = Dead_Zone_Hinge_OC_per_sample
        if implementation == "hinge_full_grad":
            # same as`hinge`
            impl = Hinge_OC_per_sample
        if implementation == "hinge_damped":
            # `hinge` but gradients for condensation points are maximally damped
            impl = Hinge_OC_per_sample_damped
        if implementation == "hinge_manhatten":
            impl = Hinge_Manhatten_OC_per_sample
        if implementation == "hinge_qmin":
            impl = Hinge_OC_per_sample_learnable_qmin
        if implementation == "hinge_qmin_betascale_pos":
            impl = Hinge_OC_per_sample_learnable_qmin_betascale_position
        self.implementation = implementation

        # configuration here, no need for all that stuff below
        # as far as the OC part is concerned (still config for payload though)
        self.oc_loss_object = OC_loss(
            loss_impl=impl,
            q_min=q_min,
            s_b=s_b,
            use_mean_x=use_average_cc_pos,
            spect_supp=1.0,
        )
        #### the latter needs to be cleaned up

        self.energy_loss_weight = energy_loss_weight
        self.use_energy_weights = use_energy_weights
        self.train_energy_correction = train_energy_correction
        self.q_min = q_min
        self.noise_q_min = noise_q_min
        self.no_beta_norm = no_beta_norm
        self.potential_scaling = potential_scaling
        self.repulsion_scaling = repulsion_scaling
        self.s_b = s_b
        self.position_loss_weight = position_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.timing_loss_weight = timing_loss_weight
        self.use_spectators = use_spectators
        self.beta_loss_scale = beta_loss_scale
        self.use_average_cc_pos = use_average_cc_pos
        self.payload_rel_threshold = payload_rel_threshold
        self.rel_energy_mse = rel_energy_mse
        self.smooth_rep_loss = smooth_rep_loss
        self.pre_train = pre_train
        self.huber_energy_scale = huber_energy_scale
        self.downweight_low_energy = downweight_low_energy
        self.n_ccoords = n_ccoords
        self.energy_den_offset = energy_den_offset
        self.noise_scaler = noise_scaler
        self.too_much_beta_scale = too_much_beta_scale
        self.cont_beta_loss = cont_beta_loss
        self.log_energy = log_energy
        self.n_classes = n_classes
        self.prob_repulsion = prob_repulsion
        self.phase_transition = phase_transition
        self.phase_transition_double_weight = phase_transition_double_weight
        self.alt_potential_norm = alt_potential_norm
        self.payload_beta_gradient_damping_strength = (
            payload_beta_gradient_damping_strength
        )
        self.payload_beta_clip = payload_beta_clip
        self.kalpha_damping_strength = kalpha_damping_strength
        self.cc_damping_strength = cc_damping_strength
        self.beta_gradient_damping = beta_gradient_damping
        self.alt_energy_loss = alt_energy_loss
        self.repulsion_q_min = repulsion_q_min
        self.super_repulsion = super_repulsion
        self.use_local_distances = use_local_distances
        self.energy_weighted_qmin = energy_weighted_qmin
        self.super_attraction = super_attraction
        self.div_repulsion = div_repulsion
        self.dynamic_payload_scaling_onset = dynamic_payload_scaling_onset
        self.alt_energy_weight = alt_energy_weight
        self.loc_time = time.time()
        self.call_count = 0
        self.beta_push = beta_push

        assert kalpha_damping_strength >= 0.0 and kalpha_damping_strength <= 1.0

        if standard_configuration is not None:
            raise NotImplementedError("Not implemented yet")

    def calc_energy_weights(self, t_energy, t_pid=None, upmouns=True):
        return _calc_energy_weights(t_energy, t_pid, upmouns, self.alt_energy_weight)

    def softclip(self, toclip, startclipat, softness=0.1):
        assert softness > 0 and softness < 1.0
        toclip /= startclipat
        soft = softness * tf.math.log((toclip - (1.0 - softness)) / softness) + 1.0
        toclip = tf.where(toclip > 1, soft, toclip)
        toclip *= startclipat
        return toclip

    def calc_energy_correction_factor_loss(
        self,
        t_energy,
        t_dep_energies,
        pred_energy,
        pred_energy_low_quantile,
        pred_energy_high_quantile,
        return_concat=False,
    ):

        if self.energy_loss_weight == 0.0:
            return (
                pred_energy**2
                + pred_energy_low_quantile**2
                + pred_energy_high_quantile**2,
                pred_energy_high_quantile**2,
            )

        ediff = (t_energy - pred_energy * t_dep_energies) / tf.sqrt(
            tf.abs(t_energy) + 1e-3
        )

        ediff = tf.debugging.check_numerics(ediff, "eloss ediff")

        eloss = None
        if self.huber_energy_scale > 0:
            eloss = huber(ediff, self.huber_energy_scale)
        else:
            eloss = tf.math.log(ediff**2 + 1.0 + 1e-5)

        # eloss = self.softclip(eloss, 0.4)
        t_energy = tf.clip_by_value(t_energy, 0.0, 1e12)
        t_dep_energies = tf.clip_by_value(t_dep_energies, 0.0, 1e12)

        # calculate energy quantiles

        # do not propagate the gradient for quantiles further up
        pred_energy = tf.stop_gradient(pred_energy)

        corrtruth = tf.math.divide_no_nan(t_energy, t_dep_energies + 1e-3)

        print("corrtruth", tf.reduce_mean(corrtruth))
        corrtruth = tf.where(corrtruth > 5.0, 5.0, corrtruth)  # remove outliers
        corrtruth = tf.where(corrtruth < 0.2, 0.2, corrtruth)
        resolution = 1 - pred_energy / corrtruth
        l_low = resolution - pred_energy_low_quantile
        l_high = resolution - pred_energy_high_quantile
        low_energy_tau = 0.16
        high_energy_tau = 0.84
        euncloss = quantile(l_low, low_energy_tau) + quantile(l_high, high_energy_tau)

        euncloss = tf.debugging.check_numerics(euncloss, "euncloss loss")
        eloss = tf.debugging.check_numerics(eloss, "eloss loss")

        if return_concat:
            return tf.concat([eloss, euncloss], axis=-1)  # for ragged map flat values

        return eloss, euncloss

    def calc_energy_loss(self, t_energy, pred_energy):

        # FIXME: this is just for debugging
        # return (t_energy-pred_energy)**2
        eloss = 0

        t_energy = tf.clip_by_value(t_energy, 1e-4, 1e12)

        if self.huber_energy_scale > 0:
            l = tf.abs(t_energy - pred_energy)
            sqrt_t_e = tf.sqrt(t_energy + 1e-3)
            l = tf.math.divide_no_nan(
                l, tf.sqrt(t_energy + 1e-3) + self.energy_den_offset
            )
            eloss = huber(l, sqrt_t_e * self.huber_energy_scale)
        elif self.alt_energy_loss:
            ediff = t_energy - pred_energy
            l = tf.math.log(ediff**2 / (t_energy + 1e-3) + 1.0)
            eloss = l
        else:
            eloss = tf.math.divide_no_nan(
                (t_energy - pred_energy) ** 2, (t_energy + 1e-3)
            )

        eloss = self.softclip(eloss, 0.2)
        eloss = tf.debugging.check_numerics(eloss, "eloss loss")
        return eloss

    def calc_qmin_weight(self, hitenergy):
        if not self.energy_weighted_qmin:
            return self.q_min

    def calc_position_loss(self, t_pos, pred_pos):
        if tf.shape(pred_pos)[-1] == 2:  # also has z component, but don't use it here
            t_pos = t_pos[:, 0:2]
        if not self.position_loss_weight:
            return 0.0 * tf.reduce_sum((pred_pos - t_pos) ** 2, axis=-1, keepdims=True)
        # reduce risk of NaNs
        ploss = huber(
            tf.sqrt(
                tf.reduce_sum((t_pos - pred_pos) ** 2, axis=-1, keepdims=True) / (10**2)
                + 1e-2
            ),
            10.0,
        )  # is in cm
        ploss = tf.debugging.check_numerics(ploss, "ploss loss")
        return ploss  # self.softclip(ploss, 3.)

    def calc_timing_loss(self, t_time, pred_time, pred_time_unc, t_dep_energy=None):
        if self.timing_loss_weight == 0.0:
            return pred_time**2 + pred_time_unc**2

        pred_time_unc = tf.nn.relu(pred_time_unc)  # safety

        tloss = (
            tf.math.divide_no_nan((t_time - pred_time) ** 2, (pred_time_unc**2 + 1e-1))
            + pred_time_unc**2
        )
        tloss = tf.debugging.check_numerics(tloss, "tloss loss")
        if t_dep_energy is not None:
            tloss = tf.where(t_dep_energy < 1.0, 0.0, tloss)

        return tloss

    def calc_classification_loss(
        self, orig_t_pid, pred_id, t_is_unique=None, hasunique=None
    ):

        if self.classification_loss_weight <= 0:
            return tf.reduce_mean(pred_id, axis=1, keepdims=True)

        pred_id = tf.clip_by_value(pred_id, 1e-9, 1.0 - 1e-9)
        t_pid = tf.clip_by_value(orig_t_pid, 1e-9, 1.0 - 1e-9)
        classloss = tf.keras.losses.categorical_crossentropy(t_pid, pred_id)
        classloss = tf.where(
            orig_t_pid[:, -1] > 0.0, 0.0, classloss
        )  # remove ambiguous, last class flag

        # take out undefined
        classloss = tf.where(tf.reduce_sum(t_pid, axis=1) > 1.0, 0.0, classloss)
        classloss = tf.where(tf.reduce_sum(t_pid, axis=1) < 1.0 - 1e-3, 0.0, classloss)

        classloss = tf.debugging.check_numerics(classloss, "classloss")

        return classloss[
            ..., tf.newaxis
        ]  # self.softclip(classloss[...,tf.newaxis], 2.)#for high weights

    def calc_beta_push(self, betas, tidx):
        if self.beta_push <= 0.0:
            return tf.reduce_mean(betas * 0.0)  # dummy

        nonoise = tf.where(tidx >= 0, betas * 0.0 + 1.0, betas * 0.0)
        nnonoise = tf.reduce_sum(nonoise)
        bpush = tf.nn.relu(self.beta_push - betas)
        bpush = tf.math.log(bpush / self.beta_push + 0.1) ** 2
        bsum = tf.reduce_sum(bpush * nonoise)
        l = tf.math.divide_no_nan(bsum, nnonoise + 1e-2)
        l = tf.debugging.check_numerics(l, "calc_beta_push loss")
        return l  # goes up to 0.1

    def loss(self, inputs):

        assert len(inputs) == 21 or len(inputs) == 20
        hasunique = False
        if len(inputs) == 21:
            (
                pred_beta,
                pred_ccoords,
                pred_distscale,
                pred_energy,
                pred_energy_low_quantile,
                pred_energy_high_quantile,
                pred_pos,
                pred_time,
                pred_time_unc,
                pred_id,
                rechit_energy,
                t_idx,
                t_energy,
                t_pos,
                t_time,
                t_pid,
                t_spectator_weights,
                t_fully_contained,
                t_rec_energy,
                t_is_unique,
                rowsplits,
            ) = inputs
            hasunique = True
        elif len(inputs) == 20:
            (
                pred_beta,
                pred_ccoords,
                pred_distscale,
                pred_energy,
                pred_energy_low_quantile,
                pred_energy_high_quantile,
                pred_pos,
                pred_time,
                pred_time_unc,
                pred_id,
                rechit_energy,
                t_idx,
                t_energy,
                t_pos,
                t_time,
                t_pid,
                t_spectator_weights,
                t_fully_contained,
                t_rec_energy,
                rowsplits,
            ) = inputs

            t_is_unique = tf.concat([t_idx[0:1] * 0 + 1, t_idx[1:] * 0], axis=0)
            hasunique = False
            print("WARNING. functions using unique will not work as expected")

            # guard

        if rowsplits.shape[0] is None:
            return tf.constant(0, dtype="float32")

        energy_weights = self.calc_energy_weights(t_energy, t_pid)
        if not self.use_energy_weights:
            energy_weights = tf.zeros_like(energy_weights) + 1.0

        # reduce weight on not fully contained showers
        energy_weights = tf.where(
            t_fully_contained > 0, energy_weights, energy_weights * 0.01
        )

        # also kill any gradients for zero weight
        energy_loss, energy_quantiles_loss = None, None
        if self.train_energy_correction:
            energy_loss, energy_quantiles_loss = (
                self.calc_energy_correction_factor_loss(
                    t_energy,
                    t_rec_energy,
                    pred_energy,
                    pred_energy_low_quantile,
                    pred_energy_high_quantile,
                )
            )
            energy_loss *= self.energy_loss_weight
            energy_quantiles_loss *= self.energy_loss_weight
        else:
            energy_loss = self.energy_loss_weight * self.calc_energy_loss(
                t_energy, pred_energy
            )
            _, energy_quantiles_loss = self.calc_energy_correction_factor_loss(
                t_energy,
                t_rec_energy,
                pred_energy,
                pred_energy_low_quantile,
                pred_energy_high_quantile,
            )
        # energy_quantiles_loss *= self.energy_loss_weight/2.

        position_loss = self.position_loss_weight * self.calc_position_loss(
            t_pos, pred_pos
        )
        timing_loss = self.timing_loss_weight * self.calc_timing_loss(
            t_time, pred_time, pred_time_unc, t_rec_energy
        )
        classification_loss = (
            self.classification_loss_weight
            * self.calc_classification_loss(t_pid, pred_id, t_is_unique, hasunique)
        )

        full_payload = tf.concat(
            [
                energy_loss,
                position_loss,
                timing_loss,
                classification_loss,
                energy_quantiles_loss,
            ],
            axis=-1,
        )

        if self.payload_beta_clip > 0:
            full_payload = tf.where(
                pred_beta < self.payload_beta_clip, 0.0, full_payload
            )
            # clip not weight, so there is no gradient to push below threshold!

        is_spectator = t_spectator_weights  # not used right now
        # and likely never again (if the truth remains ok)
        if is_spectator is None:
            is_spectator = tf.zeros_like(pred_beta)

        # just go with it
        # full_payload = tf.debugging.check_numerics(full_payload,"full_payload has nans of infs")
        # pred_ccoords = tf.debugging.check_numerics(pred_ccoords,"pred_ccoords has nans of infs")
        # energy_weights = tf.debugging.check_numerics(energy_weights,"energy_weights has nans of infs")
        # pred_beta = tf.debugging.check_numerics(pred_beta,"beta has nans of infs")
        # safe guards
        with tf.control_dependencies(
            [
                tf.assert_equal(rowsplits[-1], pred_beta.shape[0]),
                tf.assert_equal(pred_beta >= 0.0, True),
                tf.assert_equal(pred_beta <= 1.0, True),
                tf.assert_equal(is_spectator <= 1.0, True),
                tf.assert_equal(is_spectator >= 0.0, True),
            ]
        ):

            [att, rep, noise, min_b, payload, exceed_beta], mdict = self.oc_loss_object(
                beta=pred_beta,
                x=pred_ccoords,
                d=pred_distscale,
                pll=full_payload,
                truth_idx=t_idx,
                object_weight=energy_weights,
                is_spectator_weight=is_spectator,
                rs=rowsplits,
                energies=rechit_energy,
            )

        # log the OC metrics dict, if any
        self.wandb_log(mdict)

        att *= self.potential_scaling
        rep *= self.potential_scaling * self.repulsion_scaling
        min_b *= self.beta_loss_scale
        noise *= self.noise_scaler
        exceed_beta *= self.too_much_beta_scale

        energy_loss = payload[0]
        pos_loss = payload[1]
        time_loss = payload[2]
        class_loss = payload[3]
        energy_unc_loss = payload[4]

        nan_unc = tf.reduce_any(tf.math.is_nan(energy_unc_loss))
        ccdamp = (
            self.cc_damping_strength * (0.02 * tf.reduce_mean(pred_ccoords)) ** 4
        )  # gently keep them around 0

        lossval = (
            att
            + rep
            + min_b
            + noise
            + energy_loss
            + energy_unc_loss
            + pos_loss
            + time_loss
            + class_loss
            + exceed_beta
            + ccdamp
        )

        bpush = self.calc_beta_push(pred_beta, t_idx)

        lossval = tf.reduce_mean(lossval) + bpush

        self.wandb_log(
            {
                self.name + "_attractive_loss": att,
                self.name + "_repulsive_loss": rep,
                self.name + "_min_beta_loss": min_b,
                self.name + "_noise_loss": noise,
                self.name + "_energy_loss": energy_loss,
                self.name + "_energy_unc_loss": energy_unc_loss,
                self.name + "_position_loss": pos_loss,
                self.name + "_time_loss": time_loss,
                self.name + "_class_loss": class_loss,
            }
        )

        self.maybe_print_loss(lossval)

        return lossval

    def get_config(self):
        config = {
            "energy_loss_weight": self.energy_loss_weight,
            "alt_energy_weight": self.alt_energy_weight,
            "use_energy_weights": self.use_energy_weights,
            "train_energy_correction": self.train_energy_correction,
            "q_min": self.q_min,
            "no_beta_norm": self.no_beta_norm,
            "potential_scaling": self.potential_scaling,
            "repulsion_scaling": self.repulsion_scaling,
            "s_b": self.s_b,
            "noise_q_min": self.noise_q_min,
            "position_loss_weight": self.position_loss_weight,
            "classification_loss_weight": self.classification_loss_weight,
            "timing_loss_weight": self.timing_loss_weight,
            "use_spectators": self.use_spectators,
            "beta_loss_scale": self.beta_loss_scale,
            "use_average_cc_pos": self.use_average_cc_pos,
            "payload_rel_threshold": self.payload_rel_threshold,
            "rel_energy_mse": self.rel_energy_mse,
            "smooth_rep_loss": self.smooth_rep_loss,
            "pre_train": self.pre_train,
            "huber_energy_scale": self.huber_energy_scale,
            "downweight_low_energy": self.downweight_low_energy,
            "n_ccoords": self.n_ccoords,
            "energy_den_offset": self.energy_den_offset,
            "noise_scaler": self.noise_scaler,
            "too_much_beta_scale": self.too_much_beta_scale,
            "cont_beta_loss": self.cont_beta_loss,
            "log_energy": self.log_energy,
            "n_classes": self.n_classes,
            "prob_repulsion": self.prob_repulsion,
            "phase_transition": self.phase_transition,
            "phase_transition_double_weight": self.phase_transition_double_weight,
            "alt_potential_norm": self.alt_potential_norm,
            "payload_beta_gradient_damping_strength": self.payload_beta_gradient_damping_strength,
            "payload_beta_clip": self.payload_beta_clip,
            "kalpha_damping_strength": self.kalpha_damping_strength,
            "cc_damping_strength": self.cc_damping_strength,
            "beta_gradient_damping": self.beta_gradient_damping,
            "repulsion_q_min": self.repulsion_q_min,
            "super_repulsion": self.super_repulsion,
            "use_local_distances": self.use_local_distances,
            "energy_weighted_qmin": self.energy_weighted_qmin,
            "super_attraction": self.super_attraction,
            "div_repulsion": self.div_repulsion,
            "dynamic_payload_scaling_onset": self.dynamic_payload_scaling_onset,
            "beta_push": self.beta_push,
            "implementation": self.implementation,
        }
        base_config = super(LLFullObjectCondensation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LLExtendedObjectCondensation(LLFullObjectCondensation):
    """
    Same as `LLFullObjectCondensation` but adds:
    * different version of particle ID
    * Energy uncertainty
    """

    def __init__(self, *args, **kwargs):
        super(LLExtendedObjectCondensation, self).__init__(*args, **kwargs)

    def calc_classification_loss(
        self, orig_t_pid, pred_id, t_is_unique=None, hasunique=None
    ):
        """
        Truth PID is not yet one-hot encoded
        Encoding:
            0:  Muon
            1:  Electron
            2:  Photon
            3:  Charged Hadron
            4:  Neutral Hadron
            5:  Ambiguous
        """
        if self.classification_loss_weight <= 0:
            return tf.reduce_mean(pred_id, axis=1, keepdims=True)

        charged_hadronic_conditions = [
            tf.abs(orig_t_pid) == 211,  # Charged Pions
            tf.abs(orig_t_pid) == 321,  # Charged Kaons
            tf.abs(orig_t_pid) == 2212,  # Protons
        ]
        neutral_hadronic_conditions = [
            tf.abs(orig_t_pid) == 130,  # Klong
            tf.abs(orig_t_pid) == 2112,  # Neutrons
        ]
        charged_hadronic_true = tf.reduce_any(charged_hadronic_conditions, axis=0)
        neutral_hadronic_true = tf.reduce_any(neutral_hadronic_conditions, axis=0)

        truth_pid_tmp = tf.zeros_like(orig_t_pid) - 1  # Initialize with -1
        truth_pid_tmp = tf.where(tf.abs(orig_t_pid) == 13, 0, truth_pid_tmp)  # Muons
        truth_pid_tmp = tf.where(
            tf.abs(orig_t_pid) == 11, 1, truth_pid_tmp
        )  # Electrons
        truth_pid_tmp = tf.where(tf.abs(orig_t_pid) == 22, 2, truth_pid_tmp)  # Photons
        truth_pid_tmp = tf.where(
            charged_hadronic_true, 3, truth_pid_tmp
        )  # Charged Had.
        truth_pid_tmp = tf.where(
            neutral_hadronic_true, 4, truth_pid_tmp
        )  # Neutral Had.
        truth_pid_tmp = tf.where(truth_pid_tmp == -1, 5, truth_pid_tmp)  # Catch rest
        truth_pid_tmp = tf.cast(truth_pid_tmp, tf.int32)

        truth_pid_onehot = tf.one_hot(truth_pid_tmp, depth=6)
        truth_pid_onehot = tf.reshape(truth_pid_onehot, (-1, 6))

        pred_id = tf.clip_by_value(pred_id, 1e-9, 1.0 - 1e-9)
        classloss = tf.keras.losses.categorical_crossentropy(truth_pid_onehot, pred_id)
        classloss = tf.debugging.check_numerics(classloss, "classloss")

        return classloss[..., tf.newaxis]

    def calc_energy_correction_factor_loss(
        self,
        t_energy,
        t_dep_energies,
        pred_energy,
        pred_uncertainty_low,
        pred_uncertainty_high,
        return_concat=False,
    ):
        """
        This loss uses a Bayesian approach to predict an energy uncertainty.
        * t_energy              -> Truth energy of shower
        * t_dep_energies        -> Sum of deposited energy IF clustered perfectly
        * pred_energy           -> Correction factor applied to energy
        * pred_uncertainty_low  -> predicted uncertainty
        * pred_uncertainty_high -> predicted uncertainty (should be equal to ...low)
        """
        t_energy = tf.clip_by_value(t_energy, 0.0, 1e12)
        t_dep_energies = tf.clip_by_value(t_dep_energies, 0.0, 1e12)
        t_dep_energies = tf.where(
            t_dep_energies / t_energy > 2.0, 2.0 * t_energy, t_dep_energies
        )
        t_dep_energies = tf.where(
            t_dep_energies / t_energy < 0.5, 0.5 * t_energy, t_dep_energies
        )

        epred = pred_energy * t_dep_energies
        sigma = (
            tf.abs(pred_uncertainty_high * t_dep_energies) + 1.0
        )  # abs is a safety measure

        # Uncertainty 'sigma' must minimize this term:
        # ln(2*pi*sigma^2) + (E_true - E-pred)^2/sigma^2
        matching_loss = (pred_uncertainty_low - pred_uncertainty_high) ** 2
        # prediction_loss = tf.math.divide_no_nan((t_energy - epred)**2, sigma**2)
        prediction_loss = tf.math.divide_no_nan((t_energy - epred) ** 2, sigma)
        # prediction_loss = huber(prediction_loss, d=2)

        uncertainty_loss = tf.math.log(sigma**2)

        prediction_loss = tf.debugging.check_numerics(
            prediction_loss, "E: prediction_loss"
        )
        uncertainty_loss = tf.debugging.check_numerics(
            uncertainty_loss, "E: uncertainty_loss"
        )
        matching_loss = tf.debugging.check_numerics(matching_loss, "E: matching_loss")
        prediction_loss = tf.clip_by_value(prediction_loss, 0, 10)
        uncertainty_loss = tf.clip_by_value(uncertainty_loss, 0, 10)

        if return_concat:
            return tf.concat(
                [prediction_loss, matching_loss + uncertainty_loss], axis=-1
            )
        else:
            return prediction_loss, uncertainty_loss + matching_loss


class LLExtendedObjectCondensation5(LLExtendedObjectCondensation):
    """
    Same as `LLExtendedObjecCondensation` but doesn't use the Huber loss in the energy prediction:
    """

    def __init__(self, *args, **kwargs):
        super(LLExtendedObjectCondensation5, self).__init__(*args, **kwargs)

    def calc_energy_weights(self, t_energy, t_pid=None, upmouns=True):
        return _calc_energy_weights_quadratic(
            t_energy, t_pid, upmouns, self.alt_energy_weight
        )

    def calc_energy_correction_factor_loss(
        self,
        t_energy,
        t_dep_energies,
        pred_energy,
        pred_uncertainty_low,
        pred_uncertainty_high,
        return_concat=False,
    ):
        """
        This loss uses a Bayesian approach to predict an energy uncertainty.
        * t_energy              -> Truth energy of shower
        * t_dep_energies        -> Sum of deposited energy IF clustered perfectly
        * pred_energy           -> Correction factor applied to energy
        * pred_uncertainty_low  -> predicted uncertainty
        * pred_uncertainty_high -> predicted uncertainty (should be equal to ...low)
        """
        t_energy = tf.clip_by_value(t_energy, 0.0, 1e12)
        t_dep_energies = tf.clip_by_value(t_dep_energies, 0.0, 1e12)
        # t_dep_energies = tf.where(t_dep_energies / t_energy > 2.0, 2.0 * t_energy, t_dep_energies)
        # t_dep_energies = tf.where(t_dep_energies / t_energy < 0.5, 0.5 * t_energy, t_dep_energies)

        epred = pred_energy * t_dep_energies
        # sigma = pred_uncertainty_high * t_dep_energies + 1.0
        sigma = tf.sqrt(t_energy)
        sigma = tf.clip_by_value(sigma, 1.0, 1e12)

        # Uncertainty 'sigma' must minimize this term:
        # ln(2*pi*sigma^2) + (E_true - E-pred)^2/sigma^2
        matching_loss = (pred_uncertainty_low) ** 2
        uncertainty_loss = (pred_uncertainty_high) ** 2
        prediction_loss = tf.math.divide_no_nan((t_energy - epred) ** 2, sigma)
        # uncertainty_loss = tf.math.log(sigma**2)
        # prediction_loss = tf.math.divide_no_nan((t_energy - epred)**2, sigma**2)

        matching_loss = tf.debugging.check_numerics(matching_loss, "matching_loss")
        prediction_loss = tf.debugging.check_numerics(prediction_loss, "matching_loss")
        uncertainty_loss = tf.debugging.check_numerics(
            uncertainty_loss, "matching_loss"
        )

        prediction_loss = tf.clip_by_value(prediction_loss, 0, 10)
        uncertainty_loss = tf.clip_by_value(uncertainty_loss, 0, 10)

        offset_abs = tf.abs(t_energy - epred)
        offset_abs_uncorrected = tf.abs(t_energy - t_dep_energies)
        offset_rel = tf.abs(t_energy - epred) / t_energy
        offset_rel_uncorrected = tf.abs(t_energy - t_dep_energies) / t_energy

        offset_abs_small = offset_abs[t_energy < 10]
        offset_abs_large = offset_abs[t_energy > 10]
        offset_abs_uncorrected_small = offset_abs_uncorrected[t_energy < 10]
        offset_abs_uncorrected_large = offset_abs_uncorrected[t_energy > 10]
        offset_rel_small = offset_rel[t_energy < 10]
        offset_rel_large = offset_rel[t_energy > 10]
        offset_rel_uncorrected_small = offset_rel_uncorrected[t_energy < 10]
        offset_rel_uncorrected_large = offset_rel_uncorrected[t_energy > 10]

        self.wandb_log(
            {
                self.name
                + "_absolut_energy_error_lowE": tf.reduce_mean(offset_abs_small),
                self.name
                + "_absolut_energy_error_highE": tf.reduce_mean(offset_abs_large),
                self.name
                + "_absolut_energy_uncorrected_error_lowE": tf.reduce_mean(
                    offset_abs_uncorrected_small
                ),
                self.name
                + "_absolut_energy_uncorrected_error_highE": tf.reduce_mean(
                    offset_abs_uncorrected_large
                ),
                self.name
                + "_relative_energy_error_lowE": tf.reduce_mean(offset_rel_small),
                self.name
                + "_relative_energy_error_lowE": tf.reduce_mean(offset_rel_small),
                self.name
                + "_relative_energy_uncorrected_error_lowE": tf.reduce_mean(
                    offset_rel_uncorrected_small
                ),
                self.name
                + "_relative_energy_uncorrected_error_highE": tf.reduce_mean(
                    offset_rel_uncorrected_large
                ),
            }
        )

        if return_concat:
            return tf.concat(
                [prediction_loss, matching_loss + uncertainty_loss], axis=-1
            )
        else:
            return prediction_loss, uncertainty_loss + matching_loss

    def loss(self, inputs):

        assert len(inputs) == 22
        hasunique = False
        if len(inputs) == 22:
            (
                pred_beta,
                pred_ccoords,
                pred_distscale,
                pred_energy,
                pred_energy_low_quantile,
                pred_energy_high_quantile,
                pred_pos,
                pred_time,
                pred_time_unc,
                pred_id,
                rechit_energy,
                t_idx,
                t_energy,
                t_pos,
                t_time,
                t_pid,
                t_spectator_weights,
                t_fully_contained,
                t_rec_energy,
                t_is_unique,
                recHitID,
                rowsplits,
            ) = inputs
            hasunique = True
        else:
            print("Input doesn't match expected input")

        if rowsplits.shape[0] is None:
            return tf.constant(0, dtype="float32")

        energy_weights = self.calc_energy_weights(t_energy, t_pid)
        if not self.use_energy_weights:
            energy_weights = tf.zeros_like(energy_weights) + 1.0

        # reduce weight on not fully contained showers
        energy_weights = tf.where(
            t_fully_contained > 0, energy_weights, energy_weights * 0.01
        )

        # also kill any gradients for zero weight
        energy_loss, energy_quantiles_loss = None, None
        if self.train_energy_correction:
            energy_loss, energy_quantiles_loss = (
                self.calc_energy_correction_factor_loss(
                    t_energy,
                    # energy_sum,
                    t_rec_energy,
                    pred_energy,
                    pred_energy_low_quantile,
                    pred_energy_high_quantile,
                )
            )
            energy_loss *= self.energy_loss_weight
            energy_quantiles_loss *= self.energy_loss_weight
        else:
            energy_loss = self.energy_loss_weight * self.calc_energy_loss(
                t_energy, pred_energy
            )
            _, energy_quantiles_loss = self.calc_energy_correction_factor_loss(
                t_energy,
                t_rec_energy,
                pred_energy,
                pred_energy_low_quantile,
                pred_energy_high_quantile,
            )
        # energy_quantiles_loss *= self.energy_loss_weight/2.

        position_loss = self.position_loss_weight * self.calc_position_loss(
            t_pos, pred_pos
        )
        timing_loss = self.timing_loss_weight * self.calc_timing_loss(
            t_time, pred_time, pred_time_unc, t_rec_energy
        )
        classification_loss = (
            self.classification_loss_weight
            * self.calc_classification_loss(t_pid, pred_id, t_is_unique, hasunique)
        )

        full_payload = tf.concat(
            [
                energy_loss,
                position_loss,
                timing_loss,
                classification_loss,
                energy_quantiles_loss,
            ],
            axis=-1,
        )

        if self.payload_beta_clip > 0:
            full_payload = tf.where(
                pred_beta < self.payload_beta_clip, 0.0, full_payload
            )
            # clip not weight, so there is no gradient to push below threshold!

        is_spectator = t_spectator_weights  # not used right now
        # and likely never again (if the truth remains ok)
        if is_spectator is None:
            is_spectator = tf.zeros_like(pred_beta)

        with tf.control_dependencies(
            [
                tf.assert_equal(rowsplits[-1], pred_beta.shape[0]),
                tf.assert_equal(pred_beta >= 0.0, True),
                tf.assert_equal(pred_beta <= 1.0, True),
                tf.assert_equal(is_spectator <= 1.0, True),
                tf.assert_equal(is_spectator >= 0.0, True),
            ]
        ):

            [att, rep, noise, min_b, payload, exceed_beta], mdict = self.oc_loss_object(
                beta=pred_beta,
                x=pred_ccoords,
                d=pred_distscale,
                pll=full_payload,
                truth_idx=t_idx,
                object_weight=energy_weights,
                is_spectator_weight=is_spectator,
                rs=rowsplits,
                energies=rechit_energy,
            )

        # log the OC metrics dict, if any
        self.wandb_log(mdict)

        att *= self.potential_scaling
        rep *= self.potential_scaling * self.repulsion_scaling
        min_b *= self.beta_loss_scale
        noise *= self.noise_scaler
        exceed_beta *= self.too_much_beta_scale

        energy_loss = payload[0]
        pos_loss = payload[1]
        time_loss = payload[2]
        class_loss = payload[3]
        energy_unc_loss = payload[4]

        nan_unc = tf.reduce_any(tf.math.is_nan(energy_unc_loss))
        ccdamp = (
            self.cc_damping_strength * (0.02 * tf.reduce_mean(pred_ccoords)) ** 4
        )  # gently keep them around 0

        lossval = (
            att
            + rep
            + min_b
            + noise
            + energy_loss
            + energy_unc_loss
            + pos_loss
            + time_loss
            + class_loss
            + exceed_beta
            + ccdamp
        )

        bpush = self.calc_beta_push(pred_beta, t_idx)

        lossval = tf.reduce_mean(lossval) + bpush

        self.wandb_log(
            {
                self.name + "_attractive_loss": att,
                self.name + "_repulsive_loss": rep,
                self.name + "_min_beta_loss": min_b,
                self.name + "_noise_loss": noise,
                self.name + "_energy_loss": energy_loss,
                self.name + "_energy_unc_loss": energy_unc_loss,
                self.name + "_position_loss": pos_loss,
                self.name + "_time_loss": time_loss,
                self.name + "_class_loss": class_loss,
            }
        )

        self.maybe_print_loss(lossval)

        return lossval


class LLExtendedObjectCondensation6(LLExtendedObjectCondensation):
    """
    Same as `LLExtendedObjecCondensation` but doesn't use the Huber loss in the energy prediction:
    """

    def __init__(self, *args, **kwargs):
        super(LLExtendedObjectCondensation6, self).__init__(*args, **kwargs)

    def calc_energy_weights(self, t_energy, t_pid=None, upmouns=True):
        return _calc_energy_weights_quadratic(
            t_energy, t_pid, upmouns, self.alt_energy_weight
        )

    def calc_energy_correction_factor_loss(
        self,
        t_energy,
        t_dep_energies,
        pred_energy,
        pred_uncertainty_low,
        pred_uncertainty_high,
        return_concat=False,
    ):
        """
        This loss uses a Bayesian approach to predict an energy uncertainty.
        * t_energy              -> Truth energy of shower
        * t_dep_energies        -> Sum of deposited energy IF clustered perfectly
        * pred_energy           -> Correction factor applied to energy
        * pred_uncertainty_low  -> predicted uncertainty
        * pred_uncertainty_high -> predicted uncertainty (should be equal to ...low)
        """
        t_energy = tf.clip_by_value(t_energy, 0.0, 1e12)
        t_dep_energies = tf.clip_by_value(t_dep_energies, 0.0, 1e12)
        # t_dep_energies = tf.where(t_dep_energies / t_energy > 2.0, 2.0 * t_energy, t_dep_energies)
        # t_dep_energies = tf.where(t_dep_energies / t_energy < 0.5, 0.5 * t_energy, t_dep_energies)

        epred = pred_energy * t_dep_energies
        sigma = pred_uncertainty_high * t_dep_energies + 1.0
        # sigma = tf.sqrt(t_energy)
        sigma = tf.clip_by_value(sigma, 1.0, 1e12)

        # Uncertainty 'sigma' must minimize this term:
        # ln(2*pi*sigma^2) + (E_true - E-pred)^2/sigma^2
        matching_loss = (pred_uncertainty_low) ** 2
        # uncertainty_loss = (pred_uncertainty_high) ** 2
        # prediction_loss = tf.math.divide_no_nan((t_energy - epred) ** 2, sigma)
        uncertainty_loss = tf.math.log(sigma**2)
        prediction_loss = tf.math.divide_no_nan((t_energy - epred)**2, sigma**2)

        matching_loss = tf.debugging.check_numerics(matching_loss, "matching_loss")
        prediction_loss = tf.debugging.check_numerics(prediction_loss, "matching_loss")
        uncertainty_loss = tf.debugging.check_numerics(
            uncertainty_loss, "matching_loss"
        )

        prediction_loss = tf.clip_by_value(prediction_loss, 0, 10)
        uncertainty_loss = tf.clip_by_value(uncertainty_loss, 0, 10)

        offset_abs = tf.abs(t_energy - epred)
        offset_abs_uncorrected = tf.abs(t_energy - t_dep_energies)
        offset_rel = tf.abs(t_energy - epred) / t_energy
        offset_rel_uncorrected = tf.abs(t_energy - t_dep_energies) / t_energy

        offset_abs_small = offset_abs[t_energy < 10]
        offset_abs_large = offset_abs[t_energy > 10]
        offset_abs_uncorrected_small = offset_abs_uncorrected[t_energy < 10]
        offset_abs_uncorrected_large = offset_abs_uncorrected[t_energy > 10]
        offset_rel_small = offset_rel[t_energy < 10]
        offset_rel_large = offset_rel[t_energy > 10]
        offset_rel_uncorrected_small = offset_rel_uncorrected[t_energy < 10]
        offset_rel_uncorrected_large = offset_rel_uncorrected[t_energy > 10]

        self.wandb_log(
            {
                self.name
                + "_absolut_energy_error_lowE": tf.reduce_mean(offset_abs_small),
                self.name
                + "_absolut_energy_error_highE": tf.reduce_mean(offset_abs_large),
                self.name
                + "_absolut_energy_uncorrected_error_lowE": tf.reduce_mean(
                    offset_abs_uncorrected_small
                ),
                self.name
                + "_absolut_energy_uncorrected_error_highE": tf.reduce_mean(
                    offset_abs_uncorrected_large
                ),
                self.name
                + "_relative_energy_error_lowE": tf.reduce_mean(offset_rel_small),
                self.name
                + "_relative_energy_error_lowE": tf.reduce_mean(offset_rel_small),
                self.name
                + "_relative_energy_uncorrected_error_lowE": tf.reduce_mean(
                    offset_rel_uncorrected_small
                ),
                self.name
                + "_relative_energy_uncorrected_error_highE": tf.reduce_mean(
                    offset_rel_uncorrected_large
                ),
            }
        )

        if return_concat:
            return tf.concat(
                [prediction_loss, matching_loss + uncertainty_loss], axis=-1
            )
        else:
            return prediction_loss, uncertainty_loss + matching_loss

    def loss(self, inputs):

        assert len(inputs) == 22
        hasunique = False
        if len(inputs) == 22:
            (
                pred_beta,
                pred_ccoords,
                pred_distscale,
                pred_energy,
                pred_energy_low_quantile,
                pred_energy_high_quantile,
                pred_pos,
                pred_time,
                pred_time_unc,
                pred_id,
                rechit_energy,
                t_idx,
                t_energy,
                t_pos,
                t_time,
                t_pid,
                t_spectator_weights,
                t_fully_contained,
                t_rec_energy,
                t_is_unique,
                recHitID,
                rowsplits,
            ) = inputs
            hasunique = True
        else:
            print("Input doesn't match expected input")

        if rowsplits.shape[0] is None:
            return tf.constant(0, dtype="float32")

        energy_weights = self.calc_energy_weights(t_energy, t_pid)
        if not self.use_energy_weights:
            energy_weights = tf.zeros_like(energy_weights) + 1.0

        # reduce weight on not fully contained showers
        energy_weights = tf.where(
            t_fully_contained > 0, energy_weights, energy_weights * 0.01
        )

        # also kill any gradients for zero weight
        energy_loss, energy_quantiles_loss = None, None
        if self.train_energy_correction:
            energy_loss, energy_quantiles_loss = (
                self.calc_energy_correction_factor_loss(
                    t_energy,
                    # energy_sum,
                    t_rec_energy,
                    pred_energy,
                    pred_energy_low_quantile,
                    pred_energy_high_quantile,
                )
            )
            energy_loss *= self.energy_loss_weight
            energy_quantiles_loss *= self.energy_loss_weight
        else:
            energy_loss = self.energy_loss_weight * self.calc_energy_loss(
                t_energy, pred_energy
            )
            _, energy_quantiles_loss = self.calc_energy_correction_factor_loss(
                t_energy,
                t_rec_energy,
                pred_energy,
                pred_energy_low_quantile,
                pred_energy_high_quantile,
            )
        # energy_quantiles_loss *= self.energy_loss_weight/2.

        position_loss = self.position_loss_weight * self.calc_position_loss(
            t_pos, pred_pos
        )
        timing_loss = self.timing_loss_weight * self.calc_timing_loss(
            t_time, pred_time, pred_time_unc, t_rec_energy
        )
        classification_loss = (
            self.classification_loss_weight
            * self.calc_classification_loss(t_pid, pred_id, t_is_unique, hasunique)
        )

        full_payload = tf.concat(
            [
                energy_loss,
                position_loss,
                timing_loss,
                classification_loss,
                energy_quantiles_loss,
            ],
            axis=-1,
        )

        if self.payload_beta_clip > 0:
            full_payload = tf.where(
                pred_beta < self.payload_beta_clip, 0.0, full_payload
            )
            # clip not weight, so there is no gradient to push below threshold!

        is_spectator = t_spectator_weights  # not used right now
        # and likely never again (if the truth remains ok)
        if is_spectator is None:
            is_spectator = tf.zeros_like(pred_beta)

        with tf.control_dependencies(
            [
                tf.assert_equal(rowsplits[-1], pred_beta.shape[0]),
                tf.assert_equal(pred_beta >= 0.0, True),
                tf.assert_equal(pred_beta <= 1.0, True),
                tf.assert_equal(is_spectator <= 1.0, True),
                tf.assert_equal(is_spectator >= 0.0, True),
            ]
        ):

            [att, rep, noise, min_b, payload, exceed_beta], mdict = self.oc_loss_object(
                beta=pred_beta,
                x=pred_ccoords,
                d=pred_distscale,
                pll=full_payload,
                truth_idx=t_idx,
                object_weight=energy_weights,
                is_spectator_weight=is_spectator,
                rs=rowsplits,
                energies=rechit_energy,
            )

        # log the OC metrics dict, if any
        self.wandb_log(mdict)

        att *= self.potential_scaling
        rep *= self.potential_scaling * self.repulsion_scaling
        min_b *= self.beta_loss_scale
        noise *= self.noise_scaler
        exceed_beta *= self.too_much_beta_scale

        energy_loss = payload[0]
        pos_loss = payload[1]
        time_loss = payload[2]
        class_loss = payload[3]
        energy_unc_loss = payload[4]

        nan_unc = tf.reduce_any(tf.math.is_nan(energy_unc_loss))
        ccdamp = (
            self.cc_damping_strength * (0.02 * tf.reduce_mean(pred_ccoords)) ** 4
        )  # gently keep them around 0

        lossval = (
            att
            + rep
            + min_b
            + noise
            + energy_loss
            + energy_unc_loss
            + pos_loss
            + time_loss
            + class_loss
            + exceed_beta
            + ccdamp
        )

        bpush = self.calc_beta_push(pred_beta, t_idx)

        lossval = tf.reduce_mean(lossval) + bpush

        self.wandb_log(
            {
                self.name + "_attractive_loss": att,
                self.name + "_repulsive_loss": rep,
                self.name + "_min_beta_loss": min_b,
                self.name + "_noise_loss": noise,
                self.name + "_energy_loss": energy_loss,
                self.name + "_energy_unc_loss": energy_unc_loss,
                self.name + "_position_loss": pos_loss,
                self.name + "_time_loss": time_loss,
                self.name + "_class_loss": class_loss,
            }
        )

        self.maybe_print_loss(lossval)

        return lossval
