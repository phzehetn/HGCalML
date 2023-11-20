"""
Flexible training script that should be mostly configured with a yaml config file
"""

import os
import pdb
import sys
import yaml
import shutil
from argparse import ArgumentParser

import wandb
from wandb_callback import wandbCallback
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout

from DeepJetCore.training.DeepJet_callbacks import simpleMetricsCallback
from DeepJetCore.DJCLayers import StopGradient, ScalarMultiply

import training_base_hgcal
from Layers import ScaledGooeyBatchNorm2
from Layers import MixWhere, DummyLayer
from Layers import RaggedGravNet
from Layers import PlotCoordinates
from Layers import DistanceWeightedMessagePassing, TranslationInvariantMP
from Layers import LLFillSpace
from Layers import LLExtendedObjectCondensation
from Layers import DictModel, KNN
from Layers import RaggedGlobalExchange
from Layers import SphereActivation, RandomOnes
from Layers import Multi
from Layers import ShiftDistance
from Layers import LLRegulariseGravNetSpace
from Layers import SplitOffTracks, ConcatRaggedTensors, SelectTracks, ScatterBackTracks
from Regularizers import AverageDistanceRegularizer
from model_blocks import tiny_pc_pool, condition_input
from model_blocks import extent_coords_if_needed
from model_blocks import create_outputs
from model_tools import apply_weights_from_path
from model_blocks import random_sampling_unit, random_sampling_block, random_sampling_block2
from noise_filter import noise_filter
from callbacks import plotClusteringDuringTraining
from callbacks import plotClusterSummary
from callbacks import NanSweeper, DebugPlotRunner

####################################################################################################
### Load Configuration #############################################################################
####################################################################################################

parser = ArgumentParser('training')
parser.add_argument('configFile')
parser.add_argument('--run_name', help="wandb run name")
CONFIGFILE = sys.argv[1]
print(f"Using config File: \n{CONFIGFILE}")

with open(CONFIGFILE, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

N_CLUSTER_SPACE_COORDINATES = config['General']['n_cluster_space_coordinates']
N_GRAVNET_SPACE_COORDINATES = config['General']['n_gravnet_space_coordinates']
GRAVNET_ITERATIONS = len(config['General']['gravnet'])
LOSS_OPTIONS = config['LossOptions']
BATCHNORM_OPTIONS = config['BatchNormOptions']
DENSE_ACTIVATION = config['DenseOptions']['activation']
DENSE_REGULARIZER = tf.keras.regularizers.l2(config['DenseOptions']['kernel_regularizer_rate'])
DROPOUT = config['DenseOptions']['dropout']


RECORD_FREQUENCY = 10
PLOT_FREQUENCY = 80

wandb_config = {
    "loss_implementation"           :   config['General']['oc_implementation'],
    "gravnet_iterations"            :   GRAVNET_ITERATIONS,
    "gravnet_space_coordinates"     :   N_GRAVNET_SPACE_COORDINATES,
    "cluster_space_coordinates"     :   N_CLUSTER_SPACE_COORDINATES,
    "loss_energy_weight"            :   config['LossOptions']['energy_loss_weight'],
    "loss_classification_weight"    :   config['LossOptions']['classification_loss_weight'],
    "loss_qmin"                     :   config['LossOptions']['q_min'],
    "loss_use_average_cc_pos"       :   config['LossOptions']['use_average_cc_pos'],
    "loss_too_much_beta_scale"      :   config['LossOptions']['too_much_beta_scale'],
    "loss_beta_scale"               :   config['LossOptions']['beta_loss_scale'],
    "batch_max_viscosity"           :   config['BatchNormOptions']['max_viscosity'],
    "dense_activation"              :   config['DenseOptions']['activation'],
    "dense_kernel_reg"              :   config['DenseOptions']['kernel_regularizer_rate'] ,
    "dense_dropout"                 :   config['DenseOptions']['dropout'],
}

for i in range(GRAVNET_ITERATIONS):
    wandb_config[f"gravnet_{i}_neighbours"] =config['General']['gravnet'][i]['n']
for i in range(len(config['Training'])):
    wandb_config[f"train_{i}_lr"] = config['Training'][i]['learning_rate']
    wandb_config[f"train_{i}_epochs"] = config['Training'][i]['epochs']
    wandb_config[f"train_{i}_batchsize"] = config['Training'][i]['batch_size']
    if i == 1:
        wandb_config[f"train_{i}+_max_visc"] = 0.999
        wandb_config[f"train_{i}+_fluidity_decay"] = 0.1

wandb.init(
    project="jans_new_playground",
    config=wandb_config,
)
wandb.save(sys.argv[0]) # Save python file
wandb.save(sys.argv[1]) # Save config file


###############################################################################
### Define Model ##############################################################
###############################################################################

USE_BATCHNORM=True
USE_KERAS_BATCHNORM=True
USE_RANDOM=False

def TEGN_block(x, rs, 
               K : int,
               trfs :list, 
               N_coords : int,
               extra = [],
               name = 'TEGN_block'):
    x_pre = x
    coords = Dense(N_coords, name = name+"_coords")(x)
    nidx, distsq = KNN(K=K)([coords, rs])
    x = TranslationInvariantMP(trfs, activation='elu',name = name+ "_te_mp", 
                               layer_norm = True,
                               sum_weight = True)([x, nidx, distsq])
    for i,e in enumerate(extra):
        x = Dense(e, activation='elu',name = name+ f"_extra_dense_{i}")(x)
    x = Dense(x_pre.shape[1], activation='elu',name = name+ "_out_dense")(x)
    return tf.keras.layers.Add()([x, x_pre]), coords, nidx, distsq #this makes it explicitly TE

def DAF_block(x, nidx, distsq,
              prop_K_out: list = [],
              name="DAF_block"):

    from Layers import SelectFromIndicesWithPad, SortAndSelectNeighbours, RemoveSelfRef
    
    nidx = RemoveSelfRef()(nidx)
    distsq = RemoveSelfRef()(distsq)
    sdsq, sidxs = SortAndSelectNeighbours(K= nidx.shape[1], sort=True)([distsq, nidx]) #sort once then only select
    out = []
    x_in = x
    for i,d in enumerate(prop_K_out):
        assert len(d) == 3
        np, K, nout = d
        assert isinstance(np, int) and isinstance(K, int) and isinstance(nout, int)
        tdsq, tidx = SortAndSelectNeighbours(K=K,sort=False)([sdsq, sidxs])
        x_s = Dense(np, name=name+f'_pre_dense_{i}', use_bias = False)(x_in) #bias doesn't do anything
        x_s = SelectFromIndicesWithPad(subtract_self=True)([tidx,x_s]) #make it TE
        x_s = tf.keras.layers.Flatten()(x_s)
        x_s = Dense(nout, name=name+f'_post_dense_{i}', activation='elu')(x_s)
        out.append(x_s)
        x_in = Concatenate()([x_in, x_s])#add back non TE for next round
    if len(out) > 1:
        return Concatenate()(out)
    else:
        return out[0]



def config_model(Inputs, td, debug_outdir=None, plot_debug_every=RECORD_FREQUENCY*PLOT_FREQUENCY):
    """
    Function that defines the model to train
    """

    ###########################################################################
    ### Pre-processing step ###################################################
    ###########################################################################

    orig_input = td.interpretAllModelInputs(Inputs)
    pre_processed = condition_input(orig_input, no_scaling=True, no_prime=False)

    prime_coords = pre_processed['prime_coords']
    c_coords = prime_coords
    is_track = pre_processed['is_track']
    rs = pre_processed['row_splits']
    energy = pre_processed['rechit_energy']
    t_idx = pre_processed['t_idx']
    x = pre_processed['features']


    ###########################################################################
    ### the normalisation will be loaded externally, all frozen
    ###########################################################################

    x = ScaledGooeyBatchNorm2(
            fluidity_decay=0.01,
            max_viscosity=0.999999,
            name='batchnorm_tracks',
            learn=False,
            no_bias_gamma=True,
            trainable = False,
            no_gaus=False)([x, is_track])

    x = ScaledGooeyBatchNorm2(
            fluidity_decay=0.01,
            max_viscosity=0.999999,
            invert_condition=True,
            name='batchnorm_hits',
            learn=False,
            trainable = False,
            no_bias_gamma=True,
            no_gaus=False)([x, is_track])

    c_coords = prime_coords
    c_coords = ScaledGooeyBatchNorm2(
        name='batchnorm_ccoords',
        fluidity_decay=0.01,
        no_bias_gamma=True,
        trainable = False,
        max_viscosity=0.999999)(c_coords)
    
    if False: # for training the norm
        #train input norm first
        return DictModel(inputs=Inputs, outputs=[x,c_coords])
    
    ###########################################################################
    ###########################################################################
    
    c_coords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir=debug_outdir,
        name='input_c_coords',
        # publish = publishpath
        )([c_coords, energy, t_idx, rs])
    #c_coords = extent_coords_if_needed(c_coords, x, N_CLUSTER_SPACE_COORDINATES)

    x = Concatenate()([x, c_coords, is_track])
    x = Dense(64, name='dense_pre_loop', activation=DENSE_ACTIVATION)(x)

    allfeat = []
    print("Available keys: ", pre_processed.keys())

    ###########################################################################
    ### Loop over GravNet Layers ##############################################
    ###########################################################################

    rs_track, tridx = None, None

    for i in range(GRAVNET_ITERATIONS):

        d_shape = x.shape[1]//2
        x = Dense(d_shape,activation=DENSE_ACTIVATION,
            kernel_regularizer=DENSE_REGULARIZER)(x)
        if USE_BATCHNORM:
            if USE_KERAS_BATCHNORM:
                x = tf.keras.layers.BatchNormalization()(x)
            else:
                x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)
 
        use_is_track = is_track
        if rs_track is None:
            if USE_RANDOM:
                use_is_track = RandomOnes(0.02)(rs)
                use_is_track = tf.keras.layers.Add()([use_is_track, is_track])#also keep tracks
            x_track,tridx,rs_track = SelectTracks(return_rs = True)([use_is_track, x, rs])
        else:
            x_track = SelectTracks()([tridx, x])#track rs don't change, indices also the same, use only in selector mode

        #for regularisation
        pc_track = SelectTracks()([tridx, prime_coords])

        xgn_track, gncoords, gnidx, gndist = TEGN_block(x_track, rs_track, config['General']['gravnet'][i]['n'], 6*[32], #cheap
                                                       N_CLUSTER_SPACE_COORDINATES, name = f"TEGN_block_track_{i}")
        
        xdaf_track = DAF_block( Concatenate()([x_track, xgn_track]), gnidx, gndist,
              prop_K_out =  6* [[16,16,32]], # 6 hops, 16 features each, 32 out = 128: big but powerful
              name=f'DAF_block_track_{i}')
        xgn_track = Concatenate()([xgn_track, xdaf_track])#big
        
        #regularise
        gndist = LLRegulariseGravNetSpace(
                    scale=0.01,record_metrics=True,
                    name=f'regularise_gravnet_tracks_{i}')([gndist, pc_track, gnidx])
        x = DummyLayer()([x,gndist])#make sure the regulariser is not optimised away


        xgn_track = ScatterBackTracks()([use_is_track, xgn_track, tridx])
        x = Concatenate()([xgn_track, x])#now there will be zeros in places where there are no tracks
        
        #for everything
        xgn, gncoords, gnidx, gndist = TEGN_block(x, rs, config['General']['gravnet'][i]['n'], 1*[32], 
                                                       N_CLUSTER_SPACE_COORDINATES, name = f"TEGN_block_common_{i}")

        xdaf = DAF_block( Concatenate()([x, xgn]), gnidx, gndist,
              prop_K_out =  3* [[6,16,16]], # 6 hops, 4 features each, 32 out = 192: big but powerful
              name=f'DAF_block_common_{i}')
        xgn = Concatenate()([xgn, xdaf])#big

        #regularise
        gndist = LLRegulariseGravNetSpace(
                    scale=0.01,record_metrics=True,
                    name=f'regularise_gravnet_{i}')([gndist, prime_coords, gnidx])
        x = DummyLayer()([x,gndist])#make sure the regulariser is not optimised away

        x = Concatenate()([x, xgn])
        
        gncoords = PlotCoordinates(
            plot_every=plot_debug_every,
            outdir=debug_outdir,
            name='gn_coords_'+str(i)
            )([gncoords, energy, t_idx, rs])
        x = DummyLayer()([x,gncoords])#make sure the plotting is not optimised away

        if USE_BATCHNORM:
            if USE_KERAS_BATCHNORM:
                x = tf.keras.layers.BatchNormalization()(x)
            else:
                x = ScaledGooeyBatchNorm2(**BATCHNORM_OPTIONS)(x)
        x = Dense(d_shape,
                  name=f"dense_post_gravnet_1_iteration_{i}",
                  activation=DENSE_ACTIVATION,
                  kernel_regularizer=DENSE_REGULARIZER)(x)

        if USE_BATCHNORM:
            if USE_KERAS_BATCHNORM:
                x = tf.keras.layers.BatchNormalization()(x)
            else:
                x = ScaledGooeyBatchNorm2(name=f"batchnorm_loop1_iteration_{i}",**BATCHNORM_OPTIONS)(x)

        allfeat.append(x)

        if len(allfeat) > 1:
            x = Concatenate()(allfeat)
        else:
            x = allfeat[0]

    ###########################################################################
    ### Create output of model and define loss ################################
    ###########################################################################


    x = Dense(64,
              name=f"dense_final_{1}",
              activation=DENSE_ACTIVATION,
              kernel_regularizer=DENSE_REGULARIZER)(x)
    x = Dense(64,
              name=f"dense_final_{2}",
              activation=DENSE_ACTIVATION,
              kernel_regularizer=DENSE_REGULARIZER)(x)
    x = Dense(64,
              name=f"dense_final_{3}",
              activation=DENSE_ACTIVATION,
              kernel_regularizer=DENSE_REGULARIZER)(x)
    if USE_BATCHNORM:
        if USE_KERAS_BATCHNORM:
            x = tf.keras.layers.BatchNormalization()(x)
        else:
            x = ScaledGooeyBatchNorm2(name=f"batchnorm_final",**BATCHNORM_OPTIONS)(x)

    pred_beta, pred_ccoords, pred_dist, \
        pred_energy_corr, pred_energy_low_quantile, pred_energy_high_quantile, \
        pred_pos, pred_time, pred_time_unc, pred_id = \
        create_outputs(x, n_ccoords=N_CLUSTER_SPACE_COORDINATES, fix_distance_scale=True,
                is_track=is_track,
                set_track_betas_to_one=True
                )

    # pred_ccoords = LLFillSpace(maxhits=2000, runevery=5, scale=0.01)([pred_ccoords, rs, t_idx])

    if config['General']['oc_implementation'] == 'hinge':
        loss_implementation = 'hinge'
    else:
        loss_implementation = ''

    pred_beta = LLExtendedObjectCondensation(scale=1.,
                                             use_energy_weights=True,
                                             record_metrics=True,
                                             print_loss=True,
                                             print_batch_time=True,
                                             name="ExtendedOCLoss",
                                             implementation = loss_implementation,
                                             **LOSS_OPTIONS)(
        [pred_beta, pred_ccoords, pred_dist, pred_energy_corr, pred_energy_low_quantile,
         pred_energy_high_quantile, pred_pos, pred_time, pred_time_unc, pred_id, energy,
         pre_processed['t_idx'] , pre_processed['t_energy'] , pre_processed['t_pos'] ,
         pre_processed['t_time'] , pre_processed['t_pid'] , pre_processed['t_spectator_weight'],
         pre_processed['t_fully_contained'], pre_processed['t_rec_energy'],
         pre_processed['t_is_unique'], pre_processed['row_splits']])

    pred_ccoords = PlotCoordinates(
        plot_every=plot_debug_every,
        outdir = debug_outdir,
        name='condensation'
        )([pred_ccoords, pred_beta, pre_processed['t_idx'], rs])

    model_outputs = {
        'pred_beta': pred_beta,
        'pred_ccoords': pred_ccoords,
        'pred_energy_corr_factor': pred_energy_corr,
        'pred_energy_low_quantile': pred_energy_low_quantile,
        'pred_energy_high_quantile': pred_energy_high_quantile,
        'pred_pos': pred_pos,
        'pred_time': pred_time,
        'pred_id': pred_id,
        'pred_dist': pred_dist,
        'rechit_energy': energy,
        'row_splits': pre_processed['row_splits'],
        # 'no_noise_sel': pre_processed['no_noise_sel'],
        # 'no_noise_rs': pre_processed['no_noise_rs'],
        }

    return DictModel(inputs=Inputs, outputs=model_outputs)


###############################################################################
### Set up training ###########################################################
###############################################################################


train = training_base_hgcal.HGCalTraining(parser=parser)

if not train.modelSet():
    train.setModel(
        config_model,
        td=train.train_data.dataclass(),
        debug_outdir=train.outputDir+'/intplots',
        )
    #train.setCustomOptimizer(tf.keras.optimizers.Nadam())#clipnorm=1.))
    train.compileModel(learningrate=1e-4)
    #get pre norm layers
    apply_weights_from_path(os.getenv("HGCALML")+"/models/pre_norm/model.h5", train.keras_model)


    train.keras_model.summary()

###############################################################################
### Callbacks #################################################################
###############################################################################


samplepath = train.val_data.getSamplePath(train.val_data.samples[0])
PUBLISHPATH = ""
PUBLISHPATH += [d  for d in train.outputDir.split('/') if len(d)][-1]

cb = [NanSweeper()] #this takes a bit of time checking each batch but could be worth it
cb += [
    plotClusteringDuringTraining(
        use_backgather_idx=8 + i,
        outputfile=train.outputDir + "/localclust/cluster_" + str(i) + '_',
        samplefile=samplepath,
        after_n_batches=500,
        on_epoch_end=False,
        publish=None,
        use_event=0
        )
    for i in [0, 2, 4]
    ]

cb += [
    simpleMetricsCallback(
        output_file=train.outputDir+'/metrics.html',
        record_frequency= RECORD_FREQUENCY,
        plot_frequency = PLOT_FREQUENCY,
        select_metrics=[
            'ExtendedOCLoss_loss',
            'ExtendedOCLoss_dynamic_payload_scaling',
            'ExtendedOCLoss_attractive_loss',
            'ExtendedOCLoss_repulsive_loss',
            'ExtendedOCLoss_min_beta_loss',
            'ExtendedOCLoss_noise_loss',
            'ExtendedOCLoss_class_loss',
            'ExtendedOCLoss_energy_loss',
            'ExtendedOCLoss_energy_unc_loss',
            # 'ExtendedOCLoss_time_std',
            # 'ExtendedOCLoss_time_pred_std',
            '*regularise_gravnet_*',
            '*_gravReg*',
            '*containment*',
            '*contamination*',
            ],
        publish=PUBLISHPATH #no additional directory here (scp cannot create one)
        ),
    ]

cb += [


    simpleMetricsCallback(
        output_file=train.outputDir+'/val_metrics.html',
        call_on_epoch=True,
        select_metrics='val_*',
        publish=PUBLISHPATH #no additional directory here (scp cannot create one)
        ),
    ]

cb += [
    plotClusterSummary(
        outputfile=train.outputDir + "/clustering/",
        samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
        after_n_batches=1000
        )
    ]

cb += [wandbCallback()]

###############################################################################
### Actual Training ###########################################################
###############################################################################

shutil.copyfile(CONFIGFILE, os.path.join(sys.argv[3], "config.yaml"))

N_TRAINING_STAGES = len(config['Training'])
for i in range(N_TRAINING_STAGES):
    print(f"Starting training stage {i}")
    learning_rate = config['Training'][i]['learning_rate']
    epochs = config['Training'][i]['epochs']
    batch_size = config['Training'][i]['batch_size']
    train.change_learning_rate(learning_rate)
    print(f"Training for {epochs} epochs")
    print(f"Learning rate set to {learning_rate}")
    print(f"Batch size: {batch_size}")

    if i == 1:
        # change batchnorm
        for layer in train.keras_model.layers:
            if 'batchnorm' in layer.name:
                layer.max_viscosity = 0.999
                layer.fluidity_decay = 0.01
    model, history = train.trainModel(
        nepochs=epochs,
        batchsize=batch_size,
        additional_callbacks=cb
        )
