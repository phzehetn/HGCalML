import pdb
import numpy as np
import tensorflow as tf


def create_alpha_ids(pred_sid, pred_beta, is_track=None):
    pass



def process_endcap_partial(
        hits2showers_layer,
        energy_gather_layer,
        features_dict,
        predictions_dict,
        is_minbias=None):
    """
    Almost identical to `process_endcap`.
    Difference is that this takes into account the existence of tracks when
    summing over the energies.
    As we don't have the `is_track` variable included in the features or
    predictions we currently identify tracks over their z-position (Z==315)

    If there is no `no_noise_sel` in the predictions dict, it will be assumed
    that no noise filtering has been done/is necessary.
    """
    N_pred = len(predictions_dict['pred_beta'])
    predictions_dict['no_noise_sel'] = np.arange(N_pred).reshape((N_pred,1)).astype(int)
    is_track = np.abs(features_dict['recHitZ']) == 315

    pred_sid, _, alpha_idx, what, ncond = hits2showers_layer(
            predictions_dict['pred_ccoords'],
            predictions_dict['pred_beta'],
            predictions_dict['pred_dist'])
    alpha_idx_hits = tf.cast(alpha_idx, tf.float32)
    alpha_idx_tracks = tf.cast(alpha_idx, tf.float32)
    pred_sid = pred_sid.numpy()

    processed_pred_dict = dict()
    processed_pred_dict['pred_sid'] = pred_sid

    return processed_pred_dict, alpha_idx


def process_endcap(
        hits2showers_layer,
        energy_gather_layer,
        features_dict,
        predictions_dict,
        is_minbias=None):
    """
    Almost identical to `process_endcap`.
    Difference is that this takes into account the existence of tracks when
    summing over the energies.
    As we don't have the `is_track` variable included in the features or
    predictions we currently identify tracks over their z-position (Z==315)

    If there is no `no_noise_sel` in the predictions dict, it will be assumed
    that no noise filtering has been done/is necessary.
    """
    N_pred = len(predictions_dict['pred_beta'])
    predictions_dict['no_noise_sel'] = np.arange(N_pred).reshape((N_pred,1)).astype(int)
    is_track = np.abs(features_dict['recHitZ']) == 315

    pred_sid, _, alpha_idx, what, ncond = hits2showers_layer(
            predictions_dict['pred_ccoords'],
            predictions_dict['pred_beta'],
            predictions_dict['pred_dist'])
    alpha_idx_hits = tf.cast(alpha_idx, tf.float32)
    alpha_idx_tracks = tf.cast(alpha_idx, tf.float32)
    pred_sid = pred_sid.numpy()

    processed_pred_dict = dict()
    processed_pred_dict['pred_sid'] = pred_sid

    energy_data = OCGatherNumpy(
            pred_sid,
            predictions_dict,
            features_dict,
            )
    processed_pred_dict['pred_energy'] = energy_data['pred_energy']
    processed_pred_dict['pred_energy_hits_raw'] = energy_data['pred_energy_hits']
    processed_pred_dict['pred_energy_hits_corrected'] = energy_data['pred_energy_hits_corrected']
    processed_pred_dict['pred_energy_tracks_raw'] = energy_data['pred_energy_tracks']
    processed_pred_dict['pred_energy_tracks_corrected'] = energy_data['pred_energy_tracks_corrected']
    processed_pred_dict['pred_energy_beta'] = energy_data['pred_energies_beta']
    processed_pred_dict['pred_energy_beta_corrected'] = energy_data['pred_energies_beta_corrected']
    processed_pred_dict['pred_energy_cutoff'] = energy_data['pred_energy_cutoff']
    processed_pred_dict['energy_track_hit_ratio'] = energy_data['energy_track_hit_ratio']
    processed_pred_dict['condensate_hit'] = energy_data['condensate_hit']
    processed_pred_dict['n_hits'] = energy_data['n_hits']
    processed_pred_dict['n_tracks'] = energy_data['n_tracks']
    processed_pred_dict['predicted_pid'] = energy_data['predicted_pid']
    # processed_pred_dict['pred_energy'] = energy_data['prefer_tracks_corrected'].numpy()
    # processed_pred_dict['pred_energy_hits_raw'] = energy_data['hits_raw'].numpy()
    # processed_pred_dict['pred_energy_hits_corrected'] = energy_data['hits_corrected'].numpy()
    # processed_pred_dict['pred_energy_tracks_raw'] = energy_data['tracks_raw'].numpy()
    # processed_pred_dict['pred_energy_tracks_corrected'] = energy_data['tracks_corrected'].numpy()

    return processed_pred_dict, alpha_idx


def OCGatherNumpy(pred_sid, predictions_dict, features_dict):
    """
    Inputs: 
        pred_sid            -> Shower IDs returned from hits2showers_layer
        predictions_dict    -> Various useful properties
        features_dict       -> other useful properties
    """
    beta = predictions_dict['pred_beta']
    corr_factor = predictions_dict['pred_energy_corr_factor']
    pred_pid = predictions_dict['pred_id']
    is_track = features_dict['recHitID']
    energy = features_dict['recHitEnergy']
    
    # All of this code is for one event
    MULTI_TRACK = 0
    # Number of showers that include more than one track
    ENERGIES_HITS = np.zeros_like(beta)
    # Sum of all hits
    ENERGIES_HITS_CORRECTED = np.zeros_like(beta)
    # Sum of all hits x Correction factor of highest beta hit
    ENERGIES_TRACKS = np.zeros_like(beta)
    # Sum of all tracks
    ENERGIES_TRACKS_CORRECTED = np.zeros_like(beta)
    # Sum of all tracks x Correction factor of highest beta track
    ENERGIES_BETA = np.zeros_like(beta)
    # Either `TRACKS` or `HITS` depending
    # on which has the higher value for beta
    ENERGIES_BETA_CORRECTED = np.zeros_like(beta)
    # Either `TRACKS_CORRECTED` or `HITS_CORRECTED`
    # depending on which has the higher value for beta
    ENERGY_RATIO = np.zeros_like(beta)
    # Ratio between (uncorrected) track-energy vs uncorrected hit-energy
    ENERGIES_CUTOFF = np.zeros_like(beta)
    # This should be `ENERGIES_BETA_CORRECTED` as long as track energy
    # doesn't deviate too much from the sum of the hits. (e.g. ~20%).
    # The reasoning behind this is that matching over IoU otherwise
    # will lead to problems when low-energy tracks are part of
    # high-energy clusters
    CONDENSATE_HIT = np.zeros_like(beta)
    # Boolean, True if the condensation point is a hit (instead of track)
    N_HITS = np.zeros_like(beta)
    # Number of hits in each shower
    N_TRACKS = np.zeros_like(beta)
    # Number of tracks in each shower
    PRED_PID = np.zeros_like(beta)
    prepro = dict()
    for sid in np.unique(pred_sid):
        mask = pred_sid == sid
        mask_hit = np.logical_and(mask, is_track == 0)
        mask_track = np.logical_and(mask, is_track == 1)
        beta_hit = np.where(mask_hit, beta, 0.)
        beta_track = np.where(mask_track, beta, 0.)
        n_hits = np.sum(mask_hit)
        n_tracks = np.sum(mask_track)
        if n_hits == 0:
            correction_hit = 1.
            beta_max_hit = 0.
        else:
            correction_hit = corr_factor[np.argmax(beta_hit)]
            beta_max_hit = np.max(beta_hit)
        if n_tracks == 0:
            correction_track = 1.
            beta_max_track = 0.0
        else:
            correction_track = corr_factor[np.argmax(beta_track)]
            beta_max_track = np.max(beta_track)
        if n_tracks > 1:
            MULTI_TRACK += 1
        condensate_hit = beta_max_hit > beta_max_track
        hit_sum = np.sum(energy[mask_hit])
        track_sum = np.sum(energy[mask_track])
        energy_ratio = hit_sum / track_sum

        if condensate_hit:
            energies_beta = hit_sum
            energies_beta_corrected = correction_hit * hit_sum
            pid = np.argmax(pred_pid[np.argmax(beta_hit)])
        else:
            energies_beta = track_sum
            energies_beta_corrected = correction_track * track_sum
            pid = np.argmax(pred_pid[np.argmax(beta_track)])
        if np.abs(energy_ratio - 1.) < 0.2:
            ENERGIES_CUTOFF = np.where(mask, energies_beta_corrected, ENERGIES_CUTOFF)
        else:
            ENERGIES_CUTOFF = np.where(mask, correction_hit * hit_sum, ENERGIES_CUTOFF)

        PRED_PID = np.where(mask, pid, PRED_PID)
        ENERGY_RATIO = np.where(mask, energy_ratio, ENERGY_RATIO)
        ENERGIES_HITS = np.where(mask, hit_sum, ENERGIES_HITS)
        ENERGIES_HITS_CORRECTED = np.where(mask, correction_hit * hit_sum, ENERGIES_HITS_CORRECTED)
        ENERGIES_TRACKS = np.where(mask, track_sum, ENERGIES_TRACKS)
        ENERGIES_TRACKS_CORRECTED = np.where(mask, correction_track * track_sum, ENERGIES_TRACKS_CORRECTED)
        ENERGIES_BETA = np.where(mask, energies_beta, ENERGIES_BETA)
        ENERGIES_BETA_CORRECTED = np.where(mask, energies_beta_corrected, ENERGIES_BETA_CORRECTED)
        N_HITS = np.where(mask, n_hits, N_HITS)
        N_TRACKS = np.where(mask, n_tracks, N_TRACKS)
        CONDENSATE_HIT = np.where(mask, condensate_hit, CONDENSATE_HIT)

    prepro['pred_energy'] = ENERGIES_CUTOFF # Chosen as default
    prepro['pred_energy_hits'] = ENERGIES_HITS
    prepro['pred_energy_hits_corrected'] = ENERGIES_HITS_CORRECTED
    prepro['pred_energy_tracks'] = ENERGIES_TRACKS
    prepro['pred_energy_tracks_corrected'] = ENERGIES_TRACKS_CORRECTED
    prepro['pred_energies_beta'] = ENERGIES_BETA
    prepro['pred_energies_beta_corrected'] = ENERGIES_BETA_CORRECTED
    prepro['pred_energy_cutoff'] = ENERGIES_CUTOFF
    prepro['energy_track_hit_ratio'] = ENERGY_RATIO
    prepro['condensate_hit'] = CONDENSATE_HIT
    prepro['n_hits'] = N_HITS
    prepro['n_tracks'] = N_TRACKS
    prepro['predicted_pid'] = PRED_PID
    return prepro


class OCGatherEnergyLoop(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OCGatherEnergyLoop, self).__init__(**kwargs)


    # @tf.function
    def call(self,
            pred_sid,
            predictions_dict,
            features_dict):

        is_track = tf.cast(features_dict['recHitID'], bool)
        rechit_energy = features_dict['recHitEnergy']                           # (N_hits, 1)
        pred_beta = predictions_dict['pred_beta']
        pred_corr = predictions_dict['pred_energy_corr_factor']
        energy_raw_hits = tf.zeros_like(rechit_energy, tf.float32)
        energy_corrected_hits = tf.zeros_like(rechit_energy, tf.float32)
        energy_raw_tracks = tf.zeros_like(rechit_energy, tf.float32)
        energy_corrected_tracks = tf.zeros_like(rechit_energy, tf.float32)

        n_showers = tf.reduce_max(pred_sid)+1
        for shower_id in range(n_showers):
            mask = pred_sid == shower_id
            mask_hit = tf.logical_and(mask, tf.logical_not(is_track))
            mask_track = tf.logical_and(mask, is_track)
            beta_hit = tf.where(
                    tf.logical_and(mask, tf.math.logical_not(is_track)),
                    tf.cast(pred_beta, tf.float32),
                    tf.zeros_like(pred_beta, tf.float32))
            beta_track = tf.where(
                    tf.logical_and(mask, is_track),
                    tf.cast(pred_beta, tf.float32),
                    tf.zeros_like(pred_beta, tf.float32))
            if tf.reduce_sum(tf.cast(mask_hit, tf.float32)) > 0:
                correction_hit = tf.gather(pred_corr, tf.argmax(beta_hit))
                # print(correction_hit)
            else:
                correction_hit = 0.
            if tf.reduce_sum(tf.cast(mask_track, tf.float32)) > 0:
                correction_track = tf.gather(pred_corr, tf.argmax(beta_track))
            else:
                correction_track = 0.
            hit_indices = tf.where(tf.reshape(mask_hit, (-1,)))
            track_indices = tf.where(tf.reshape(mask_track, (-1,)))
            energy_raw_hit = tf.reduce_sum(tf.gather_nd(rechit_energy, hit_indices))
            energy_raw_track = tf.reduce_sum(tf.gather_nd(rechit_energy, track_indices))

            energy_corrected_hit = tf.reshape(
                    correction_hit * energy_raw_hit, (-1,))
            energy_corrected_track = tf.reshape(
                    correction_track * energy_raw_track, (-1,))

            indices = tf.where(mask)
            updates_e_raw_hit = tf.broadcast_to(
                    energy_raw_hit, [tf.shape(indices)[0],])
            updates_e_corrected_hit = tf.broadcast_to(
                    energy_corrected_hit, [tf.shape(indices)[0],])
            updates_e_raw_track = tf.broadcast_to(
                    energy_raw_track, [tf.shape(indices)[0],])
            updates_e_corrected_track = tf.broadcast_to(
                    energy_corrected_track, [tf.shape(indices)[0],])
            energy_raw_hits = tf.tensor_scatter_nd_update(
                    energy_raw_hits, indices, updates_e_raw_hit)
            energy_corrected_hits = tf.tensor_scatter_nd_update(
                    energy_corrected_hits, indices, updates_e_corrected_hit)
            energy_raw_tracks = tf.tensor_scatter_nd_update(
                    energy_raw_tracks, indices, updates_e_raw_track)
            energy_corrected_tracks = tf.tensor_scatter_nd_update(
                    energy_corrected_tracks, indices, updates_e_corrected_track)

        e_prefer_tracks_raw = tf.where(energy_raw_tracks != 0,
                energy_raw_tracks, energy_raw_hits)
        e_prefer_tracks_corrected = tf.where(energy_corrected_tracks != 0,
                energy_corrected_tracks, energy_corrected_hits)
        e_prefer_hits_raw = tf.where(energy_raw_hits != 0,
                energy_raw_hits, energy_raw_tracks)
        e_prefer_hits_corrected = tf.where(energy_corrected_hits != 0,
                energy_corrected_hits, energy_corrected_tracks)

        data = {
            'tracks_raw': energy_raw_tracks,
            'tracks_corrected': energy_corrected_tracks,
            'hits_raw': energy_raw_hits,
            'hits_corrected': energy_corrected_hits,
            'prefer_tracks_raw': e_prefer_tracks_raw,
            'prefer_tracks_corrected': e_prefer_tracks_corrected,
            'prefer_hits_raw': e_prefer_hits_raw,
            'prefer_hits_corrected': e_prefer_hits_corrected,
            }

        return data


class OCGatherEnergy(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OCGatherEnergy, self).__init__(**kwargs)


    def call(self,
            pred_sid,
            predictions_dict,
            features_dict,
            alpha_idx):

        sid = pred_sid + 1
        is_track = features_dict['recHitID']
        predicted_noise = sid == 0                                              # (N_hits, 1)
        corr_factor = tf.where(                                                 # [N_hits, 1]
                tf.math.logical_or(predicted_noise, is_track),
                tf.zeros_like(predictions_dict['pred_energy_corr_factor'], tf.float32),
                tf.cast(predictions_dict['pred_energy_corr_factor'], tf.float32))
        rechit_energy = features_dict['recHitEnergy']                           # (N_hits, 1)

        alpha_corr_hit = tf.gather_nd(corr_factor, alpha_idx[:,tf.newaxis])     # [N_showers, 1]
        alpha_corr_with_noise_hit = tf.concat(
                [tf.zeros([1,1], dtype=tf.float32), alpha_corr_hit], axis=0)        # [N_showers+1, 1]

        e_shower_hit = tf.math.unsorted_segment_sum(                                # [N_showers+1,] 
            rechit_energy[:,0], sid[:,0], 
            num_segments=(tf.reduce_max(sid)+1))
        e_hit_shower_corrected = e_shower_hit * alpha_corr_with_noise_hit[:,0]  # [N_showers+1,] 
        e_hit_raw = tf.reshape(                                                 # [N_hits, 1]
            tf.gather(e_shower_hit, sid[:,0]),
            shape=(-1,1))
        e_hit_cor = tf.reshape(                                                 # [N_hits, 1]
            tf.gather(e_hit_shower_corrected, sid[:,0]),
            shape=(-1,1))

        data = {
            # 'tracks_raw': e_track_raw,
            # 'tracks_corrected': e_track_corrected,
            'hits_raw': e_hit_raw,
            'hits_corrected': e_hit_cor,
            # 'no_minbias_fraction': no_minbias_fraction,
            }

        print(f"Energy remaining: {tf.reduce_sum(rechit_energy)}")
        return data


class OCGatherEnergyCorrFac(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OCGatherEnergyCorrFac, self).__init__(**kwargs)

    def call(self,
            pred_sid,
            predictions_dict,
            features_dict,
            alpha_idx_tracks,
            alpha_idx_hits):
        """
        Same as `OCGatherEnergyCorrFac` with the addition that one can chose if the energy
        reconstructed with the tracks or the enery reconstructed by the calorimeter should be used.
        Shapes:
            * rechit_energy, is_track:      [N_orig, 1]     (before noise filter)
            * pred_sid, pred_corr_factor:   [N_filtered, 1] (after noise filter)
            * pred_beta:                    [N_filtered, 1] (after noise filter)
            * no_noise_idx:                 [N_filtered, 1] (with indices up to N_orig as entries)
        Returns:
            Will always return only one value:
                return_tracks_where_possible overrides `return_tracks`
            By default returns hit energy
        """

        pred_sid_p1 = pred_sid +1 # Take care of -1 label for noise
        pred_corr_factor = tf.where(
                pred_sid==-1,
                tf.zeros_like(predictions_dict['pred_energy_corr_factor'], tf.float32),
                tf.cast(predictions_dict['pred_energy_corr_factor'], tf.float32))
        rechit_energy = tf.where(
                pred_sid==-1,
                tf.zeros_like(features_dict['recHitEnergy'], tf.float32),
                tf.cast(features_dict['recHitEnergy'], tf.float32))

        is_track = features_dict['recHitID']
        e_hit = tf.where(is_track==0, rechit_energy, tf.zeros_like(rechit_energy))
        e_track = tf.where(is_track==1, rechit_energy, tf.zeros_like(rechit_energy))
        pred_beta = predictions_dict['pred_beta']
        beta_hit = tf.where(is_track==0, pred_beta, tf.zeros_like(pred_beta))
        beta_track = tf.where(is_track==1, pred_beta, tf.zeros_like(pred_beta))

        e_hit_shower = tf.math.unsorted_segment_sum(
            e_hit[:,0], pred_sid_p1[:,0], 
            num_segments=(tf.reduce_max(pred_sid_p1)+1))
        e_track_shower = tf.math.unsorted_segment_sum(
            e_track[:,0], pred_sid_p1[:,0], 
            num_segments=(tf.reduce_max(pred_sid_p1)+1))

        zero_appendix = tf.constant([[0]], dtype=pred_corr_factor.dtype)
        pred_corr_factor = tf.concat(
                [pred_corr_factor, zero_appendix], axis=0)
        """
        alpha_idx_hits = tf.where(
                tf.math.is_nan(tf.cast(alpha_idx_hits, dtype=tf.float32)),
                -1.0,
                alpha_idx_hits)
        alpha_idx_tracks = tf.where(
                tf.math.is_nan(tf.cast(alpha_idx_tracks, dtype=tf.float32)),
                -1.0,
                alpha_idx_tracks)
        """
        # TODO: The next line doesn't work, so we have to do another way to select the hits
        # correction_hits = pred_corr_factor[tf.cast(alpha_idx_hits, dtype=tf.int32)]
        correction_hits = tf.gather_nd(
                pred_corr_factor,
                tf.cast(alpha_idx_hits, tf.int32)[:,tf.newaxis])
        correction_hits = tf.concat([ [[0.]], correction_hits], axis=0)
        correction_tracks = tf.gather_nd(
                pred_corr_factor,
                tf.cast(alpha_idx_tracks, tf.int32)[:,tf.newaxis])
        correction_tracks = tf.concat([ [[0.]], correction_tracks], axis=0)
        # correction_tracks = pred_corr_factor[tf.cast(alpha_idx_tracks, dtype=tf.int32)]
        e_hit_shower_corrected = e_hit_shower * correction_hits[:,0]
        e_track_shower_corrected = e_track_shower * correction_tracks[:,0]
        e_hit_corrected = tf.reshape(
                tf.gather(e_hit_shower_corrected, pred_sid_p1[:,0]),
                shape=(-1,1))
        e_hit_raw = tf.reshape(
                tf.gather(e_hit_shower, pred_sid_p1[:,0]),
                shape=(-1,1))
        e_track_corrected = tf.reshape(
                tf.gather(e_track_shower_corrected, pred_sid_p1[:,0]),
                shape=(-1,1))
        e_track_raw = tf.reshape(
                tf.gather(e_track_shower, pred_sid_p1[:,0]),
                shape=(-1,1))


        data = {
            'tracks_raw': e_track_raw,
            'tracks_corrected': e_track_corrected,
            'hits_raw': e_hit_raw,
            'hits_corrected': e_hit_corrected,
            # 'no_minbias_fraction': no_minbias_fraction,
            }

        return data

