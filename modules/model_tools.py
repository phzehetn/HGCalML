import tensorflow as tf
from Layers import RaggedGravNet
from DeepJetCore.customObjects import get_custom_objects
from DeepJetCore.modeltools import apply_weights_where_possible
import numpy as np


# debug overwrite
def apply_weights_where_possible(
    target_model, weight_model, strict=True, match_remaining_by_type_and_shape=False
):

    layers_set = []

    def try_to_set_weights(layer_a, layer_b):
        try:
            layer_a.set_weights(layer_b.get_weights())
            print("using weights from ", layer_a.name)
            layers_set.append(layer_a.name)
        except Exception as e:
            print(e)
            print(
                "target shape\n",
                [a.shape for a in layer_a.get_weights()],
                "\nsource shape:\n",
                [a.shape for a in layer_b.get_weights()],
            )
            if strict:
                print("strict mode, raising exception")
                raise e

    for layer_a in target_model.layers:
        for layer_b in weight_model.layers:
            if layer_a.name == layer_b.name:
                if isinstance(layer_a, RaggedGravNet):  # THIS IS A COMPATIBILITY HACK!!
                    try_to_set_weights(
                        layer_a.input_feature_transform, layer_b.input_feature_transform
                    )
                    try_to_set_weights(
                        layer_a.input_spatial_transform, layer_b.input_spatial_transform
                    )
                    try_to_set_weights(
                        layer_a.output_feature_transform,
                        layer_b.output_feature_transform,
                    )
                    layers_set.append(layer_a.name)
                else:
                    try_to_set_weights(layer_a, layer_b)

    if match_remaining_by_type_and_shape:
        for layer_a in target_model.layers:
            if layer_a.name in layers_set:
                continue
            for layer_b in weight_model.layers:
                if layer_b.name in layers_set:
                    continue
                if layer_a.__class__.__name__ == layer_b.__class__.__name__:
                    if len(layer_a.get_weights()) == len(layer_b.get_weights()):
                        # if all shapes are the same
                        shapes_a = [a.shape for a in layer_a.get_weights()]
                        shapes_b = [a.shape for a in layer_b.get_weights()]
                        if all([a == b for a, b in zip(shapes_a, shapes_b)]):
                            try_to_set_weights(layer_a, layer_b)
                            layers_set.append(layer_a.name)
    # print all layers that were *not* set
    for layer_a in target_model.layers:
        if layer_a.name not in layers_set:
            print("WARNING: layer", layer_a.name, "was not set")

    return target_model


def apply_weights_from_path(
    path_to_weight_model,
    existing_model,
    return_weight_model=False,
    apply_optimizer=False,
    strict=True,
):

    weightmodel = tf.keras.models.load_model(
        path_to_weight_model, custom_objects=get_custom_objects()
    )
    existing_model = apply_weights_where_possible(existing_model, weightmodel, strict)

    try:
        for le, lw in zip(existing_model.layers, weightmodel.layers):
            if hasattr(le, "get_config"):
                if le.get_config() != lw.get_config():
                    ce = le.get_config()
                    cw = lw.get_config()
                    for cek in ce.keys():
                        if ce[cek] != cw[cek]:
                            print(
                                "warning: different configuration",
                                le.name,
                                ":",
                                cek,
                                ce[cek],
                                cw[cek],
                            )
    except:
        pass
    if apply_optimizer:
        existing_model.optimizer = weightmodel.optimizer
    if return_weight_model:
        return existing_model, weightmodel
    return existing_model


def _issame(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, (list,)):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not _issame(a[i], b[i]):
                return False
        return True
    if isinstance(a, (np.ndarray,)):
        return np.all(a == b)
    return a == b


def apply_and_freeze_common_weights(path_to_weight_model, existing_model, keep_open=[]):

    existing_model, weightmodel = apply_weights_from_path(
        path_to_weight_model, existing_model, return_weight_model=True
    )

    for le in existing_model.layers:
        for lw in weightmodel.layers:
            if not le.name == lw.name:
                continue
            is_in_open = False
            for kp in keep_open:
                if kp in le.name:
                    is_in_open = True
                    break
            if (not is_in_open) and _issame(le.get_weights(), lw.get_weights()):
                print("freezing", le.name)
                le.trainable = False
            else:
                print("not freezing", le.name)

    return existing_model
