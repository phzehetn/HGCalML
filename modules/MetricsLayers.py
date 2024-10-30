import tensorflow as tf
from baseModules import LayerWithMetrics
from GravNetLayersRagged import SelectFromIndices as SelIdx


class MLBase(LayerWithMetrics):
    def __init__(self, active=True, **kwargs):
        """
        Output is always inputs[0]
        Use active=False to mask truth inputs in the inheriting class.
        No need to activate or deactivate metrics recording (done in LayerWithMetrics)

        Inherit and implement metrics_call method

        """
        self.active = active
        super(MLBase, self).__init__(**kwargs)

    def get_config(self):
        config = {"active": self.active}
        base_config = super(MLBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shapes):
        return input_shapes[0]

    def build(self, input_shapes):
        super(MLBase, self).build(input_shapes)

    def metrics_call(self, inputs):
        pass

    def call(self, inputs):
        if self.active:
            self.metrics_call(inputs)
        return inputs[0]


class MLReductionMetrics(MLBase):

    def __init__(self, **kwargs):
        """
        Inputs:
        - neighbour group selection indices (V_red x 1)
        - truth indices (V x 1)
        - truth energies (V x 1)
        - row splits (for V)
        - sel row splits (for V_red)

        """
        self.extended = False
        super(MLReductionMetrics, self).__init__(**kwargs)

    def metrics_call(self, inputs):
        istrack = None
        other_feat = None
        if len(inputs) == 5:
            gsel, tidx, ten, rs, srs = inputs
        if len(inputs) == 6:
            gsel, tidx, ten, istrack, rs, srs = inputs
        if len(inputs) == 7:
            gsel, tidx, ten, istrack, other_feat, rs, srs = inputs
        # tf.assert_equal(tidx.shape,ten.shape)#safety

        alltruthcount = None
        seltruthcount = None
        nonoisecounts_bef = []
        nonoisecounts_after = []

        if rs.shape[0] is None:
            return

        stidx, sten = tf.constant([[0]], dtype="int32"), tf.constant(
            [[0.0]], dtype="float32"
        )
        n_track_before, n_track_after = tf.constant(
            [[0.0]], dtype="float32"
        ), tf.constant([[0.0]], dtype="float32")

        if self.active:
            stidx, sten = SelIdx.raw_call(gsel, [tidx, ten])
            for i in tf.range(rs.shape[0] - 1):
                u, _, c = tf.unique_with_counts(tidx[rs[i] : rs[i + 1], 0])
                nonoisecounts_bef.append(c[u >= 0])
                if alltruthcount is None:
                    alltruthcount = u.shape[0]
                else:
                    alltruthcount += u.shape[0]

                u, _, c = tf.unique_with_counts(stidx[srs[i] : srs[i + 1], 0])
                nonoisecounts_after.append(c[u >= 0])
                if seltruthcount is None:
                    seltruthcount = u.shape[0]
                else:
                    seltruthcount += u.shape[0]

            if istrack is not None:
                n_track_before = tf.reduce_sum(istrack)
                n_track_after = SelIdx.raw_call(gsel, [istrack])
                n_track_after = tf.reduce_sum(n_track_after)

        nonoisecounts_bef = tf.concat(nonoisecounts_bef, axis=0)
        nonoisecounts_after = tf.concat(nonoisecounts_after, axis=0)

        lostfraction = 1.0 - tf.cast(seltruthcount, dtype="float32") / (
            tf.cast(alltruthcount, dtype="float32")
        )

        # done with fractions
        # create the indices of one unique hit (first dimension) entry for each object (defined by tidx or unique ten)
        # that is lost after applying the selection gsel
        def create_lost_indices(tidx, ten, gsel):
            no_noise_sel = tidx[:, 0] >= 0
            ue, _, c = tf.unique_with_counts(tf.boolean_mask(ten, no_noise_sel)[:, 0])
            # ue = ue[c>3] #don't count <4 hit showers
            no_noise_sel = stidx[:, 0] >= 0
            uesel, _, c = tf.unique_with_counts(
                tf.boolean_mask(sten, no_noise_sel)[:, 0]
            )

        # for simplicity assume that no energy is an exact duplicate (definitely good enough here)
        no_noise_sel = tidx[:, 0] >= 0
        ue, _, c = tf.unique_with_counts(tf.boolean_mask(ten, no_noise_sel)[:, 0])
        # ue = ue[c>3] #don't count <4 hit showers
        no_noise_sel = stidx[:, 0] >= 0
        uesel, _, c = tf.unique_with_counts(
            tf.boolean_mask(sten, no_noise_sel)[:, 0]
        )  # only non-noise

        tot_lost_en_sum = tf.reduce_sum(ue) - tf.reduce_sum(uesel)

        allen = tf.concat([ue, uesel], axis=0)
        ue, _, c = tf.unique_with_counts(allen)

        lostenergies = ue[c < 2]

        l_em = tf.reduce_mean(lostenergies)
        l_em = tf.where(tf.math.is_finite(l_em), l_em, 0.0)

        l_ema = tf.reduce_max(lostenergies)
        l_ema = tf.where(tf.math.is_finite(l_ema), l_ema, 0.0)

        reduced_to_fraction = tf.cast(srs[-1], dtype="float32") / tf.cast(
            rs[-1], dtype="float32"
        )

        no_noise_hits_bef = tf.cast(tf.math.count_nonzero(tidx + 1), dtype="float32")
        no_noise_hits_aft = tf.cast(tf.math.count_nonzero(stidx + 1), dtype="float32")
        self.wandb_log(
            {
                self.name
                + "_hits_pobj_bef_mean": tf.reduce_mean(
                    tf.cast(nonoisecounts_bef, "float32")
                ),
                self.name + "_hits_pobj_bef_max": tf.reduce_max(nonoisecounts_bef),
                self.name
                + "_hits_pobj_after_mean": tf.reduce_mean(
                    tf.cast(nonoisecounts_after, "float32")
                ),
                self.name + "_hits_pobj_after_max": tf.reduce_max(nonoisecounts_after),
                self.name + "_lost_energy_mean": l_em,
                self.name + "_lost_energy_max": l_ema,
                self.name + "_lost_energy_sum": tot_lost_en_sum,
                self.name + "_reduction": reduced_to_fraction,
                self.name
                + "_no_noise_reduction": (no_noise_hits_aft / no_noise_hits_bef),
                self.name + "_lost_objects": lostfraction,
            }
        )

        if istrack is not None:
            self.wandb_log(
                {
                    self.name + "_tracks_bef": n_track_before,
                    self.name + "_tracks_after": n_track_after,
                }
            )
