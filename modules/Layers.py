"""
Needs some cleaning.
On the longer term, let's keep this just a wrapper module for layers,
but the layers themselves to other files
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {}

from GraphCondensationLayers import graph_condensation_layers
from tensorflow.keras.layers import LeakyReLU
from baseModules import LayerWithMetrics
from MetricsLayers import MLReductionMetrics
from Initializers import EyeInitializer
from RaggedLayers import ragged_layers


global_layers_list.update(graph_condensation_layers)
global_layers_list["LeakyReLU"] = LeakyReLU
global_layers_list["LayerWithMetrics"] = LayerWithMetrics
global_layers_list["MLReductionMetrics"] = MLReductionMetrics
global_layers_list["EyeInitializer"] = EyeInitializer
global_layers_list.update(ragged_layers)

# LayersRagged.py
from LayersRagged import (
    RaggedSumAndScatter,
    Condensate,
    CondensateToPseudoRS,
    RaggedGlobalExchange,
)

global_layers_list["RaggedSumAndScatter"] = RaggedSumAndScatter
global_layers_list["Condensate"] = Condensate
global_layers_list["CondensateToPseudoRS"] = CondensateToPseudoRS
global_layers_list["RaggedGlobalExchange"] = RaggedGlobalExchange

# GravNetLayersRagged.py
from GravNetLayersRagged import (
    CastRowSplits,
    Where,
    PrintMeanAndStd,
    ProcessFeatures,
    NeighbourCovariance,
    LocalDistanceScaling,
    SelectFromIndices,
    MultiBackGather,
    MultiBackScatter,
    KNN,
    SortAndSelectNeighbours,
    RaggedGravNet,
    TranslationInvariantMP,
    DistanceWeightedMessagePassing,
    XYZtoXYZPrime,
    RandomSampling,
)

global_layers_list["CastRowSplits"] = CastRowSplits
global_layers_list["Where"] = Where
global_layers_list["PrintMeanAndStd"] = PrintMeanAndStd
global_layers_list["ProcessFeatures"] = ProcessFeatures
global_layers_list["NeighbourCovariance"] = NeighbourCovariance
global_layers_list["LocalDistanceScaling"] = LocalDistanceScaling
global_layers_list["SelectFromIndices"] = SelectFromIndices
global_layers_list["MultiBackGather"] = MultiBackGather
global_layers_list["MultiBackScatter"] = MultiBackScatter
global_layers_list["KNN"] = KNN
global_layers_list["SortAndSelectNeighbours"] = SortAndSelectNeighbours
global_layers_list["RaggedGravNet"] = RaggedGravNet
global_layers_list["TranslationInvariantMP"] = TranslationInvariantMP
global_layers_list["DistanceWeightedMessagePassing"] = DistanceWeightedMessagePassing
global_layers_list["XYZtoXYZPrime"] = XYZtoXYZPrime
global_layers_list["RandomSampling"] = RandomSampling

# DebugLayers.py
from DebugLayers import (
    PlotCoordinates,
    PlotGraphCondensationEfficiency,
)

global_layers_list["PlotCoordinates"] = PlotCoordinates
global_layers_list["PlotGraphCondensationEfficiency"] = PlotGraphCondensationEfficiency


# LossLayers.py
from LossLayers import (
    NormaliseTruthIdxs,
    LossLayerBase,
    LLDummy,
    LLValuePenalty,
    CreateTruthSpectatorWeights,
    LLRegulariseGravNetSpace,
    LLFillSpace,
    LLClusterCoordinates,
    LLFullObjectCondensation,
    LLExtendedObjectCondensation,
    LLExtendedObjectCondensation5,
)

global_layers_list["NormaliseTruthIdxs"] = NormaliseTruthIdxs
global_layers_list["LossLayerBase"] = LossLayerBase
global_layers_list["LLDummy"] = LLDummy
global_layers_list["LLValuePenalty"] = LLValuePenalty
global_layers_list["CreateTruthSpectatorWeights"] = CreateTruthSpectatorWeights
global_layers_list["LLRegulariseGravNetSpace"] = LLRegulariseGravNetSpace
global_layers_list["LLFillSpace"] = LLFillSpace
global_layers_list["LLClusterCoordinates"] = LLClusterCoordinates
global_layers_list["LLFullObjectCondensation"] = LLFullObjectCondensation
global_layers_list["LLExtendedObjectCondensation"] = LLExtendedObjectCondensation
global_layers_list["LLExtendedObjectCondensation5"] = LLExtendedObjectCondensation5


class DummyLayer(tf.keras.layers.Layer):
    """
    Just to make sure other layers are not optimised away
    Inputs:
    - list of tensors. First will be passed through, the other will be ignored
    """

    def call(self, inputs):
        return inputs[0]


class OnesLike(Layer):
    def __init__(self, **kwargs):
        super(OnesLike, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return tf.ones_like(inputs)


class ZerosLike(Layer):
    def __init__(self, **kwargs):
        super(ZerosLike, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        return tf.zeros_like(inputs)


global_layers_list["DummyLayer"] = DummyLayer
global_layers_list["OnesLike"] = OnesLike
global_layers_list["ZerosLike"] = ZerosLike
