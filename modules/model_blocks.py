import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Concatenate,
    Add,
    BatchNormalization,
)

from DeepJetCore.DJCLayers import SelectFeatures, ScalarMultiply, StopGradient
from datastructures.TrainData_NanoML import n_id_classes

from LossLayers import LLClusterCoordinates, LLRegulariseGravNetSpace, LLValuePenalty
from GravNetLayersRagged import (
    XYZtoXYZPrime,
    RaggedGravNet,
    DistanceWeightedMessagePassing,
    KNN,
    LocalDistanceScaling,
    Where,
    CastRowSplits,
    ProcessFeatures,
)
from Initializers import EyeInitializer
from Layers import OnesLike, ZerosLike, CreateTruthSpectatorWeights
from Layers import TranslationInvariantMP, DummyLayer
from DebugLayers import PlotCoordinates, PlotGraphCondensationEfficiency
from GraphCondensationLayers import (
    GraphCondensation,
    CreateGraphCondensation,
    PushUp,
    SelectUp,
    LLGraphCondensationEdges,
    InsertEdgesIntoTransition,
    MLGraphCondensationMetrics,
    LLGraphCondensationScore,
    AddNeighbourDiff,
)


def extent_coords_if_needed(
    coords, x, n_cluster_space_coordinates, name="coord_extend"
):
    if n_cluster_space_coordinates > 3:
        x = Concatenate()([coords, x])
        extendcoords = Dense(
            n_cluster_space_coordinates - 3,
            use_bias=False,
            name=name + "_dense",
            kernel_initializer=EyeInitializer(stddev=0.001),
        )(x)
        coords = Concatenate()([coords, extendcoords])
    return coords


# new format!
def create_outputs(
    x,
    n_ccoords=3,
    n_classes=n_id_classes,
    n_pos=2,
    fix_distance_scale=True,
    energy_factor=True,
    name_prefix="output_module",
    trainable=True,
    is_track=None,
    set_track_betas_to_one=False,
    predict_spectator_weights=False,
):
    """
    returns
        * pred_beta                     Dense(1)
        * pred_ccoords                  Dense(n_ccoords)
        * pred_dist                     1 if fix_distance_scale else 2*Dense(1)
        * pred_energy                   1 + Dense(1) if energy_factor else Dense(1)
        * pred_energy_low_quantile      Dense(1)
        * pred_energy_high_quantile     Dense(1)
        * pred_pos                      Dense(n_pos)
        * pred_time                     10 + Dense(1)
        * pred_time_unc                 1 + Dense(1)
        * pred_id                       Dense(n_classes)
        if predict_specator_weights:
            * pred_spectator_weights    Dense(1) with relu and L2 regularization
    """
    if not fix_distance_scale:
        print("warning: fix_distance_scale=False can lead to issues.")

    pred_beta = Dense(
        1, activation="sigmoid", name=name_prefix + "_beta", trainable=trainable
    )(x)
    if set_track_betas_to_one:
        assert is_track is not None
        pred_beta = Where()([is_track, 0.9999, pred_beta])

    if predict_spectator_weights:
        pred_spectator_weights = Dense(
            1,
            activation="sigmoid",
            name=name_prefix + "_spectator_weight",
            trainable=trainable,
        )(x)

    pred_ccoords = Dense(
        n_ccoords,
        use_bias=False,
        name=name_prefix + "_clustercoords",
        trainable=trainable,
    )(
        x
    )  # bias has no effect

    energy_act = None
    if energy_factor:
        energy_act = "elu"
    energy_res_act = "relu"
    pred_energy = Dense(
        1,
        name=name_prefix + "_energy",
        kernel_initializer="zeros",
        activation=energy_act,
        trainable=trainable,
    )(ScalarMultiply(0.01)(x))

    if energy_factor:
        pred_energy = Add(name=name_prefix + "_one_plus_energy")(
            [OnesLike()(pred_energy), pred_energy]
        )

    pred_energy_low_quantile = Dense(
        1,
        name=name_prefix + "_energy_low_quantile",
        # kernel_initializer='zeros',
        activation=energy_res_act,
        trainable=trainable,
    )(x)

    pred_energy_high_quantile = Dense(
        1,
        name=name_prefix + "_energy_high_quantile",
        # kernel_initializer='zeros',
        activation=energy_res_act,
        trainable=trainable,
    )(x)

    pred_pos = Dense(
        n_pos, use_bias=False, name=name_prefix + "_pos", trainable=trainable
    )(x)

    pred_time = Dense(1, name=name_prefix + "_time_proxy", trainable=trainable)(
        ScalarMultiply(0.01)(x)
    )
    pred_time = Add(name=name_prefix + "_time")(
        [ScalarMultiply(10.0)(OnesLike()(pred_time)), pred_time]
    )

    pred_time_unc = Dense(
        1, activation="elu", name=name_prefix + "_time_unc", trainable=trainable
    )(ScalarMultiply(0.01)(x))
    pred_time_unc = Add()(
        [pred_time_unc, OnesLike()(pred_time_unc)]
    )  # strict positive with small turn on

    pred_id = Dense(
        n_classes,
        activation="softmax",
        name=name_prefix + "_class",
        trainable=trainable,
    )(x)

    pred_dist = OnesLike()(pred_beta)
    if not fix_distance_scale:
        pred_dist = ScalarMultiply(2.0)(
            Dense(
                1, activation="sigmoid", name=name_prefix + "_dist", trainable=trainable
            )(x)
        )
        # this needs to be bound otherwise fully anti-correlated with coordates scale
    if predict_spectator_weights:
        return (
            pred_beta,
            pred_ccoords,
            pred_dist,
            pred_energy,
            pred_energy_low_quantile,
            pred_energy_high_quantile,
            pred_pos,
            pred_time,
            pred_time_unc,
            pred_id,
            pred_spectator_weights,
        )
    else:
        return (
            pred_beta,
            pred_ccoords,
            pred_dist,
            pred_energy,
            pred_energy_low_quantile,
            pred_energy_high_quantile,
            pred_pos,
            pred_time,
            pred_time_unc,
            pred_id,
        )


def condition_input(orig_inputs, no_scaling=False, no_prime=False, new_prime=False):

    if "t_spectator_weight" not in orig_inputs.keys():  # compat layer
        orig_t_spectator_weight = CreateTruthSpectatorWeights(
            threshold=5.0, minimum=1e-1, active=True
        )([orig_inputs["t_spectator"], orig_inputs["t_idx"]])
        orig_inputs["t_spectator_weight"] = orig_t_spectator_weight

    if "is_track" not in orig_inputs.keys():
        is_track = SelectFeatures(2, 3)(orig_inputs["features"])
        orig_inputs["is_track"] = Where(outputval=1.0, condition="!=0")(
            [is_track, ZerosLike()(is_track)]
        )

    if "rechit_energy" not in orig_inputs.keys():
        orig_inputs["rechit_energy"] = SelectFeatures(0, 1)(orig_inputs["features"])

    processed_features = orig_inputs["features"]
    orig_inputs["orig_features"] = orig_inputs["features"]

    # get some things to work with
    orig_inputs["row_splits"] = CastRowSplits()(orig_inputs["row_splits"])
    orig_inputs["orig_row_splits"] = orig_inputs["row_splits"]

    # coords have not been built so features not processed, so this is the first time this is called
    if "coords" not in orig_inputs.keys():
        if not no_scaling:
            processed_features = ProcessFeatures(name="precondition_process_features")(
                orig_inputs["features"]
            )

        orig_inputs["coords"] = SelectFeatures(5, 8)(processed_features)
        orig_inputs["features"] = processed_features

        # create starting point for cluster coords
        if no_prime:
            orig_inputs["prime_coords"] = SelectFeatures(5, 8)(
                orig_inputs["orig_features"]
            )
        else:
            orig_inputs["prime_coords"] = XYZtoXYZPrime(new_prime=new_prime)(
                SelectFeatures(5, 8)(orig_inputs["orig_features"])
            )

    return orig_inputs


def expand_coords_if_needed(coords, x, ndims, name, trainable):
    if coords.shape[-1] == ndims:
        return coords
    if coords.shape[-1] > ndims:
        raise ValueError("only expanding coordinates")
    return Concatenate()(
        [
            coords,
            Dense(
                ndims - coords.shape[-1],
                kernel_initializer="zeros",
                name=name,
                trainable=trainable,
            )(x),
        ]
    )


def mini_tree_create(
    score,
    coords,
    rs,
    t_idx,
    t_energy,
    is_track=None,
    K=5,
    K_loss=64,
    score_threshold=0.5,
    low_energy_cut=4.0,  # allow everything below 4 GeV to be removed
    record_metrics=False,
    trainable=False,
    name="tree_creation_0",
    always_record_reduction=True,
    cleaning_mode=False,  # changes the score loss to the cleaning loss
):
    """
    provides the loss needed and the condensation graph
    Does not contain any learnable parameters whatsoever.
    trainable only turns on the losses
    """

    if cleaning_mode:
        lcc_input = [coords, t_idx, score, score, rs]
    else:
        lcc_input = [coords, t_idx, score, rs]

    # right now this would scale with score (makes sense); could it be useful to cut that gradient?
    coords = LLClusterCoordinates(
        record_metrics=record_metrics,
        active=trainable,
        scale=0.1,  # scale this down, this will not be the main focus
        ignore_noise=True,  # this is filtered by the graph condensation anyway
        print_batch_time=False,
        specweight_to_weight=True,
        downsample=30000,  # downsample to 30k
    )(
        lcc_input
    )  # score is not affected here

    if cleaning_mode:
        score = LLValuePenalty(1.0, active=trainable)(score)
    else:
        if is_track is not None:
            orig_score = score
            score = Where()(
                [is_track, ZerosLike()(score), score]
            )  # this will prevent collapse to a single track per particle

        score = LLGraphCondensationScore(
            record_metrics=record_metrics,
            K=K_loss,
            active=trainable,
            penalty_fraction=0.5,  # doesn't matter in current implementation
            low_energy_cut=low_energy_cut,
            name=name + "_score",
        )([score, coords, t_idx, t_energy, rs])

        if is_track is not None:
            score = Where()(
                [is_track, orig_score, score]
            )  # technically doesn't matter as tracks are always promoted

    trans_a = CreateGraphCondensation(
        score_threshold=score_threshold, K=K, name=name + "_tree"
    )(score, coords, rs, always_promote=is_track)

    trans_a = MLGraphCondensationMetrics(
        name=name + "_metrics", record_metrics=record_metrics or always_record_reduction
    )(trans_a, t_idx, t_energy)

    return trans_a


def mini_tree_clustering(
    pre_inputs: dict,  # features, rechit_energy, row_splits, t_idx,
    trans_a: GraphCondensation,
    edge_dense=[64, 64, 64],
    edge_pre_nodes=32,
    remove_ambiguous=True,
    record_metrics=False,
    trainable=False,
    name="tree_clustering_0",
    produce_output=True,  # turn off for fast pretraining
):

    energy = pre_inputs["rechit_energy"]

    x = Dense(
        edge_pre_nodes,
        activation="elu",
        kernel_initializer="he_normal",
        name=name + "_enc",
        trainable=trainable,
    )(pre_inputs["features"])
    x_e = AddNeighbourDiff()(x, trans_a)

    for i, nodes in enumerate(edge_dense):
        x_e = Dense(
            nodes,
            activation="elu",
            kernel_initializer="he_normal",
            name=name + f"_edge_dense_{i}",
            trainable=trainable,
        )(x_e)

    x_e = Dense(
        trans_a.K() + 1,
        activation="sigmoid",
        name=name + "_edge_dense_final",
        trainable=trainable,
    )(x_e)

    x_e = LLGraphCondensationEdges(
        active=trainable,
        record_metrics=record_metrics,
        treat_none_same_as_noise=remove_ambiguous,
        # set n
    )(x_e, trans_a, pre_inputs["t_idx"])

    x_e = StopGradient()(x_e)  # edges are purely learned from explicit loss
    trans_a = InsertEdgesIntoTransition()(
        x_e, trans_a
    )  # this normalises them to sum=1-noise fraction

    out = {}

    if produce_output:
        # now push up, some explicit:
        # explicit_keys = ['prime_coords', 'coords', 'rechit_energy', 'features', 'is_track', 'row_splits']

        out["prime_coords"] = PushUp(add_self=True)(
            pre_inputs["prime_coords"], trans_a, weight=energy
        )
        out["coords"] = PushUp(add_self=True)(
            pre_inputs["coords"], trans_a, weight=energy
        )
        out["rechit_energy"] = PushUp(mode="sum", add_self=True)(energy, trans_a)
        ew_feat = PushUp(add_self=False)(pre_inputs["features"], trans_a, weight=energy)
        ew_t_minbias = PushUp(add_self=False)(
            tf.cast(pre_inputs["t_only_minbias"], tf.float32), trans_a, weight=energy
        )
        w_feat = PushUp(add_self=False)(pre_inputs["features"], trans_a)
        x_sel = SelectUp()(pre_inputs["features"], trans_a)
        out["features"] = Concatenate()([x_sel, ew_feat, w_feat])
        out["t_minbias_weighted"] = ew_t_minbias

        out["is_track"] = SelectUp()(pre_inputs["is_track"], trans_a)
        out["row_splits"] = trans_a["rs_up"]

        # now explicit pass through
        out["orig_features"] = pre_inputs["orig_features"]
        out["orig_row_splits"] = pre_inputs["orig_row_splits"]

        for k in pre_inputs.keys():  # pass through truth
            if "t_" == k[0:2]:
                out[k] = SelectUp()(pre_inputs[k], trans_a)

        return out, trans_a

    # this is a dummy return
    return {}, trans_a


def GravNet_plus_TEQMP(
    name,
    x,
    cprime,
    hit_energy,
    t_idx,
    rs,
    d_shape,
    n_neighbours,
    debug_outdir,
    plot_debug_every,
    debug_publish=None,
    space_reg_strength=-1.0,
    n_gn_coords=4,
    teq_nodes=[64, 32, 16, 8],
    return_coords=False,
    trainable=True,
    add_scaling=True,
):

    xgn, gncoords, gnnidx, gndist = RaggedGravNet(
        name="GravNet_" + name,  # 76929, 42625, 42625
        n_neighbours=n_neighbours,
        n_dimensions=n_gn_coords,
        n_filters=d_shape,
        n_propagate=d_shape,
        coord_initialiser_noise=1e-3,
        feature_activation="elu",
        trainable=trainable,
    )([x, rs])

    if space_reg_strength > 0:
        gndist = LLRegulariseGravNetSpace(
            name=f"gravnet_coords_reg_{name}",
            record_metrics=True,
            scale=space_reg_strength,
        )([gndist, cprime, gnnidx])

    gncoords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir=debug_outdir,
        name=f"gncoords_{name}",
        publish=debug_publish,
    )([gncoords, hit_energy, t_idx, rs])

    x = DummyLayer()(
        [x, gncoords]
    )  # just so the branch is not optimised away, anyway used further down
    x = Concatenate()([xgn, x])

    dscale = Dense(1, name=name + "_dscale", trainable=trainable)(
        x
    )  # FIXME sigmoid needs to go
    # dscale = ScalarMultiply(2.)(dscale)
    # dscale = Multiply()([dscale, dscale]) # as distances are also quadratic
    gndist = LocalDistanceScaling()([gndist, dscale])

    x = TranslationInvariantMP(
        teq_nodes,
        layer_norm=True,
        activation=None,  # layer norm takes care
        sum_weight=False,
        name=name + "_teqmp",
        trainable=trainable,
    )([x, gnnidx, gndist])

    if return_coords:
        return Concatenate()([xgn, x]), gncoords
    return Concatenate()([xgn, x])


def tree_condensation_block(
    pre_processed,
    debug_outdir="",
    plot_debug_every=-1,
    debug_publish=None,
    name="tree_condensation_block",
    trainable=False,
    record_metrics=False,
    produce_output=True,
    always_record_reduction=True,
    decouple_coords=False,
    low_energy_cut=4.0,
    remove_ambiguous=True,
    enc_nodes=32,
    gn_nodes=16,
    gn_neighbours=16,
    teq_nodes=[16, 16],
    edge_dense=[32, 16],
    edge_pre_nodes=16,
):

    prime_coords = pre_processed["prime_coords"]
    is_track = pre_processed["is_track"]
    rs = pre_processed["row_splits"]
    energy = pre_processed["rechit_energy"]
    t_idx = pre_processed["t_idx"]
    x = pre_processed["features"]

    # keeping this in check is useful, therefore tanh is actually a good choice
    x = Dense(enc_nodes, activation="tanh", name=name + "_enc", trainable=trainable)(x)
    x = Concatenate()([prime_coords, x])

    xgn, gn_coords = GravNet_plus_TEQMP(
        name + "_net",
        x,
        prime_coords,
        energy,
        t_idx,
        rs,
        gn_nodes,  # nodes
        gn_neighbours,  # neighbours
        debug_outdir,
        plot_debug_every,
        debug_publish=debug_publish,
        teq_nodes=teq_nodes,
        return_coords=True,
        trainable=trainable,
        space_reg_strength=1e-6,
    )

    x = Concatenate()([xgn, x])

    score = Dense(1, activation="sigmoid", name=name + "_score", trainable=trainable)(x)
    pre_processed["features"] = x  # pass through
    if decouple_coords:
        # prime coordinates are passed through to here (see above)
        gn_coords = Dense(
            gn_coords.shape[1],
            name=name + "_coords",
            trainable=trainable,
            use_bias=False,
        )(x)

    ud_graph = mini_tree_create(
        score,
        gn_coords,
        rs,
        t_idx,
        pre_processed["t_energy"],
        is_track=is_track,
        K=5,
        K_loss=48,
        score_threshold=0.5,
        low_energy_cut=low_energy_cut,
        record_metrics=record_metrics,
        trainable=trainable,
        name=name + "_tree_creation",
        always_record_reduction=always_record_reduction,
    )

    out = mini_tree_clustering(
        pre_processed,
        ud_graph,
        edge_dense=edge_dense,
        edge_pre_nodes=edge_pre_nodes,
        remove_ambiguous=remove_ambiguous,
        record_metrics=record_metrics,
        trainable=trainable,
        name=name + "_tree_clustering",
        produce_output=produce_output,
    )

    return out, x


def post_tree_condensation_push(
    x_in,  # before pushing as defined in the graph
    x_mix,
    graph,
    trainable=True,
    heads: int = 4,
    mix_nodes=8,
    name="post_tree_push",
):
    """
    Defines a simple block to push up learnable quantities.
    The inputs x and x_mix are *before* the push up defined by the graph object.
    The output will have dimensionality *after* the push up.
    """
    from GraphCondensationLayers import Mix

    x = x_in
    xup = []
    for i in range(heads):
        x_mix_i = Dense(
            mix_nodes,
            activation="elu",
            trainable=trainable,
            name=name + f"_int_mix_{i}",
        )(x_mix)
        x_mix_i = Mix()(x_mix_i, graph)
        # sigmoid is ok, the 'mean' takes care of what the softmax does in classic attention
        attention = Dense(
            graph["nidx_down"].shape[1],
            activation="sigmoid",
            trainable=trainable,
            name=name + f"_weights_{i}",
        )(x_mix_i)
        # add an epsilon to avoid numerical instabilities
        # this behaves very similar to standard attention now
        xm = PushUp(add_self=False, mode="mean")(x, graph, nweights=attention)
        xup.append(xm)

    if heads > 1:
        xup = Concatenate()(xup)
    else:
        xup = xup[0]
    return xup


def tree_condensation_block2(*args, **kwargs):
    # TODO: probably needs a more descriptive name
    # just define some defaults here
    return tree_condensation_block(
        *args,
        **kwargs,
        enc_nodes=128,
        gn_nodes=64,
        gn_neighbours=128,
        teq_nodes=[64, 64],
        edge_dense=[64, 32],
        edge_pre_nodes=32,
        low_energy_cut=4.0,
        name="tree_condensation_block2",
    )


def single_tree_condensation_block(
    in_dict,
    debug_outdir="",
    plot_debug_every=-1,
    name="single_tree_condensation_block",
    trainable=False,
    record_metrics=False,
    decouple_coords=False,
    pre_gravnet=True,
    debug_publish=None,
):

    if (
        pre_gravnet
    ):  # run one single 'gravnet' to gather info about best coordinates; no need to learn coordinates yet so direct implementation

        nidx, dist = KNN(16, record_metrics=record_metrics, name="pre_knn_coords")(
            [in_dict["prime_coords"], in_dict["row_splits"]]
        )

        xpre = Concatenate()([in_dict["prime_coords"], in_dict["features"]])
        xpre = Dense(16, activation="tanh", name="pre_enc", trainable=trainable)(xpre)

        xscale = Dense(1, name="pre_scale", trainable=trainable)(xpre)
        dist = LocalDistanceScaling(name="pre_scale_dist", max_scale=10.0)(
            [dist, xscale]
        )
        if True:
            xgn = TranslationInvariantMP(
                [16],
                layer_norm=True,
                activation=None,  # layer norm takes care
                sum_weight=False,
                name=name + "_pre_teqmp",
                trainable=trainable,
            )([xpre, nidx, dist])

        else:
            xgn = DistanceWeightedMessagePassing(
                [16], name=name + "_pre_dmp1", trainable=trainable
            )([xpre, nidx, dist])

        dist = StopGradient()(dist)
        in_dict["features"] = Concatenate()([xgn, in_dict["features"], dist])

    [out, graph], x_proc = tree_condensation_block(
        in_dict,
        # the latter overwrites the default arguments such that it is in training mode
        debug_outdir=debug_outdir,
        plot_debug_every=plot_debug_every,
        trainable=trainable,
        record_metrics=record_metrics,
        debug_publish=debug_publish,
        decouple_coords=decouple_coords,
        produce_output=True,
    )

    ###########################################################################
    ### Just some debug out ###################################################
    ###########################################################################

    all_out = {"no_noise_idx_stage_0": graph["sel_idx_up"]}

    in_dict["t_energy"] = PlotGraphCondensationEfficiency(
        plot_every=plot_debug_every,
        name="dc_1st_stage",
        publish=debug_publish,
        outdir=debug_outdir,
    )(in_dict["t_energy"], in_dict["t_idx"], graph)

    in_dict["t_energy"] = PlotGraphCondensationEfficiency(
        plot_every=plot_debug_every,
        name="dc_1st_stage_no_tracks",
        publish=debug_publish,
        outdir=debug_outdir,
    )(in_dict["t_energy"], in_dict["t_idx"], graph, is_track=in_dict["is_track"])

    # just to keep the plot in the loop
    graph["weights_down"] = DummyLayer()([graph["weights_down"], in_dict["t_energy"]])

    ###########################################################################
    ### Second stage ##########################################################
    ###########################################################################

    x = Concatenate()([x_proc, in_dict["features"]])
    x = BatchNormalization(name=name + "_bn0", trainable=trainable)(
        x
    )  # this might be crucial after the summing

    xadd = post_tree_condensation_push(
        x,  # pushed (with attention)
        x,  # used to mix and build attention
        graph,
        # take default here: heads = push_heads,
        name=name + "_push",
        trainable=trainable,
    )  # this gets the original vector, adds more features to be pushed

    out["features"] = Concatenate()([out["features"], xadd])
    out["features"] = BatchNormalization(name=name + "_bn1", trainable=trainable)(
        out["features"]
    )  # this might be crucial after the summing
    return out, graph, all_out


def double_tree_condensation_block(
    in_dict,
    debug_outdir="",
    plot_debug_every=-1,
    name="double_tree_condensation_block",
    trainable=False,
    record_metrics=False,
    decouple_coords=False,
    pre_gravnet=True,
    debug_publish=None,
):

    if (
        pre_gravnet
    ):  # run one single 'gravnet' to gather info about best coordinates; no need to learn coordinates yet so direct implementation

        nidx, dist = KNN(16, record_metrics=record_metrics, name="pre_knn_coords")(
            [in_dict["prime_coords"], in_dict["row_splits"]]
        )

        xpre = Concatenate()([in_dict["prime_coords"], in_dict["features"]])
        xpre = Dense(16, activation="tanh", name="pre_enc", trainable=trainable)(xpre)

        xscale = Dense(1, name="pre_scale", trainable=trainable)(xpre)
        dist = LocalDistanceScaling(name="pre_scale_dist", max_scale=10.0)(
            [dist, xscale]
        )
        if True:
            xgn = TranslationInvariantMP(
                [16],
                layer_norm=True,
                activation=None,  # layer norm takes care
                sum_weight=False,
                name=name + "_pre_teqmp",
                trainable=trainable,
            )([xpre, nidx, dist])

        else:
            xgn = DistanceWeightedMessagePassing(
                [16], name=name + "_pre_dmp1", trainable=trainable
            )([xpre, nidx, dist])

        dist = StopGradient()(dist)
        in_dict["features"] = Concatenate()([xgn, in_dict["features"], dist])

    [out, graph], x_proc = tree_condensation_block(
        in_dict,
        # the latter overwrites the default arguments such that it is in training mode
        debug_outdir=debug_outdir,
        plot_debug_every=plot_debug_every,
        trainable=trainable,
        record_metrics=record_metrics,
        debug_publish=debug_publish,
        decouple_coords=decouple_coords,
        remove_ambiguous=False,  # just redistribute energy, don't remove any
        produce_output=True,
    )

    ###########################################################################
    ### Just some debug out ###################################################
    ###########################################################################

    all_out = {"no_noise_idx_stage_0": graph["sel_idx_up"]}

    in_dict["t_energy"] = PlotGraphCondensationEfficiency(
        plot_every=plot_debug_every,
        name="dc_1st_stage",
        publish=debug_publish,
        outdir=debug_outdir,
    )(in_dict["t_energy"], in_dict["t_idx"], graph)

    in_dict["t_energy"] = PlotGraphCondensationEfficiency(
        plot_every=plot_debug_every,
        name="dc_1st_stage_no_tracks",
        publish=debug_publish,
        outdir=debug_outdir,
    )(in_dict["t_energy"], in_dict["t_idx"], graph, is_track=in_dict["is_track"])

    # just to keep the plot in the loop
    graph["weights_down"] = DummyLayer()([graph["weights_down"], in_dict["t_energy"]])

    ###########################################################################
    ### Second stage ##########################################################
    ###########################################################################

    x = Concatenate()([x_proc, in_dict["features"]])
    x = BatchNormalization(name=name + "_bn0", trainable=trainable)(
        x
    )  # this might be crucial after the summing

    xadd = post_tree_condensation_push(
        x,  # pushed (with attention)
        x,  # used to mix and build attention
        graph,
        # take default here: heads = push_heads,
        name=name + "_push",
        trainable=trainable,
    )  # this gets the original vector, adds more features to be pushed

    out["features"] = Concatenate()([out["features"], xadd])
    out["features"] = BatchNormalization(name=name + "_bn1", trainable=trainable)(
        out["features"]
    )  # this might be crucial after the summing

    [out2, graph2], x_proc2 = tree_condensation_block2(
        out,
        debug_outdir=debug_outdir,
        plot_debug_every=plot_debug_every,
        debug_publish=debug_publish,
        trainable=trainable,
        decouple_coords=decouple_coords,
        remove_ambiguous=False,  # just redistribute energy, don't remove any
        record_metrics=record_metrics,
    )

    ###########################################################################
    ### Done, now just checks and plots #######################################
    ###########################################################################

    # this is out and not out2 on purpose!
    out["t_energy"] = PlotGraphCondensationEfficiency(
        plot_every=plot_debug_every,
        name="dc_2nd_stage",
        publish=debug_publish,
        outdir=debug_outdir,
    )(out["t_energy"], out["t_idx"], graph2)

    out["t_energy"] = PlotGraphCondensationEfficiency(
        plot_every=plot_debug_every,
        name="dc_2nd_stage_no_tracks",
        publish=debug_publish,
        outdir=debug_outdir,
    )(out["t_energy"], out["t_idx"], graph2, is_track=out["is_track"])

    # make sure the above does not get optimised away
    graph2["weights_down"] = DummyLayer()([graph2["weights_down"], out["t_energy"]])

    # create indices that were removed for tracking
    survived_both_stages = SelectUp()(graph["sel_idx_up"], graph2)
    all_out["no_noise_idx_stage_1"] = graph2["sel_idx_up"]
    all_out["survived_both_stages"] = survived_both_stages

    return out2, graph2, all_out
