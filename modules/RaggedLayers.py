"""
Only 'real' ragged layers here that take as input ragged tensors and output ragged tensors
"""

import tensorflow as tf


ragged_layers = {}


class RaggedSelectFromIndices(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """

        Inputs:
        - data (V x F)
        - indices (output from RaggedCreateCondensatesIdxs)

        Output:
        - ragged output (depending on indices)

        """
        super(RaggedSelectFromIndices, self).__init__(**kwargs)

    def call(self, inputs):
        assert len(inputs) == 2
        data, idx = inputs
        return tf.gather_nd(data, idx)


ragged_layers["RaggedSelectFromIndices"] = RaggedSelectFromIndices


class RaggedMixHitAndCondInfo(tf.keras.layers.Layer):

    def __init__(self, operation="subtract", **kwargs):
        """
        Concat only supports same number of features for both

        Inputs:
        - ragged tensor of features of hits associated to condensation points: [None, None, None, F]
        - ragged tensor of features of condensation points: [None, None, F]

        Output:
        - ragged tensor where operation is applied to first input (broadcasted):
             if add or subtract: [None, None, None, F]
             if concat: [None, None, None, F+F]
        """
        assert (
            operation == "subtract"
            or operation == "add"
            or operation == "concat"
            or operation == "mult"
        )

        if operation == "subtract":
            self.operation = self._sub
        if operation == "add":
            self.operation = self._add
        if operation == "concat":
            self.operation = self._concat
        if operation == "mult":
            self.operation = self._mult
        self.operationstr = operation

        super(RaggedMixHitAndCondInfo, self).__init__(**kwargs)

    def get_config(self):
        config = {"operation": self.operationstr}
        base_config = super(RaggedMixHitAndCondInfo, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _add(self, lhs, rhs):
        return lhs + rhs

    def _sub(self, lhs, rhs):
        return lhs - rhs

    def _concat(self, lhs, rhs):
        return tf.concat([lhs, lhs * 0 + rhs], axis=-1)

    def _mult(self, lhs, rhs):
        return lhs * rhs

    def call(self, inputs):
        assert len(inputs) == 2
        hitsrt, cprt = inputs

        tf.assert_equal(
            hitsrt.shape[-1], cprt.shape[-1], self.name + " last shape must match."
        )
        tf.assert_equal(
            hitsrt.shape[0], cprt.shape[0], self.name + " first shape must match."
        )
        tf.assert_equal(
            hitsrt.shape.ndims, cprt.shape.ndims + 1, self.name + " shape issue."
        )
        tf.assert_equal(hitsrt.shape.ndims, 4, self.name + " shape issue.")
        # tf.assert_equal(tf.reduce_sum(hitsrt.shape * 0 +1), tf.reduce_sum(cprt.shape * 0 +1) + 1, self.name+" shape issue.")
        # print('hitsrt, cprt 1',hitsrt.shape, cprt.shape)
        cprt = tf.expand_dims(cprt, axis=2)
        rt = cprt
        rrt = hitsrt

        # breaks at
        # :  tf.Tensor([1 1 1 1], shape=(4,), dtype=int32) tf.Tensor([3 1 1 1], shape=(4,), dtype=int32) tf.Tensor([1392 1377 1352 1187], shape=(4,), dtype=int32)
        # so rt has wrong row splits
        try:
            return self.operation(hitsrt, cprt)
        except Exception as e:
            print(">>>")
            while hasattr(rt, "values"):
                print(
                    ": ", rrt.row_lengths(), rt.row_lengths(), rrt.values.row_lengths()
                )
                # print(':: ',rt,rrt)
                rt = rt.values
                rrt = rrt.values
            print("<<<")
            raise e


ragged_layers["RaggedMixHitAndCondInfo"] = RaggedMixHitAndCondInfo


class RaggedToFlatRS(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        simply takes [events, ragged-something, F] and returns
        [X, F] + row splits.
        Only one ragged dimension here
        """
        super(RaggedToFlatRS, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs.values, inputs.row_splits


ragged_layers["RaggedToFlatRS"] = RaggedToFlatRS


class FlatRSToRagged(tf.keras.layers.Layer):
    def __init__(self, validate=True, **kwargs):
        """
        simply takes [X, F] + row splits and returns
        [events, ragged-something, F]
        Only one ragged dimension here
        """
        super(FlatRSToRagged, self).__init__(**kwargs)
        self.validate = validate

    def get_config(self):
        config = {"validate": self.validate}
        base_config = super(FlatRSToRagged, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        return tf.RaggedTensor.from_row_splits(
            inputs[0], inputs[1], validate=self.validate
        )


ragged_layers["FlatRSToRagged"] = FlatRSToRagged
