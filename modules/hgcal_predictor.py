import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator

from OCHits2Showers import OCHits2ShowersLayer, process_endcap, OCGatherEnergyCorrFac
from ShowersMatcher import ShowersMatcher
from datastructures.TrainData_NanoML import TrainData_NanoML
from datastructures.TrainData_PreselectionNanoML import TrainData_PreselectionNanoML
from DeepJetCore.modeltools import load_model
from datastructures import TrainData_TrackML


def add_dict_to_df(df, dictionary):
    for key in dictionary.keys():
        value = dictionary[key]
        if len(value.shape) != 2:
            continue
        # if value.shape[1] == 0:
        #     print("Nothing stored in key", key)
        if value.shape[1] == 1:
            if isinstance(value, tf.Tensor):
                df[key] = value.numpy().flatten()
            elif type(value) == np.ndarray:
                df[key] = value.flatten()
            else:
                print("Unexpected type! ", value.__class__)
        elif value.shape[1] > 1:
            for j in range(value.shape[1]):
                if isinstance(value, tf.Tensor):
                    df[key+'_'+str(j)] = value[:,j].numpy().flatten()
                elif type(value) == np.ndarray:
                    df[key+'_'+str(j)] = value[:,j].flatten()
                else:
                    pass
        else:
            raise ValueError
    return df


class HGCalPredictor():
    def __init__(self, input_source_files_list, training_data_collection, predict_dir, unbuffered=False, model_path=None, max_files=4, inputdir=None):
        self.input_data_files = []
        self.inputdir = None
        self.predict_dir = predict_dir
        self.unbuffered=unbuffered
        self.max_files = max_files
        print("Using HGCal predictor class")

        ## prepare input lists for different file formats

        if input_source_files_list[-6:] == ".djcdc":
            print('reading from data collection', input_source_files_list)
            predsamples = DataCollection(input_source_files_list)
            self.inputdir = predsamples.dataDir
            for s in predsamples.samples:
                self.input_data_files.append(s)

        elif input_source_files_list[-6:] == ".djctd":
            self.inputdir = os.path.abspath(os.path.dirname(input_source_files_list))
            infile = os.path.basename(input_source_files_list)
            self.input_data_files.append(infile)
        else:
            print('reading from text file', input_source_files_list)
            self.inputdir = os.path.abspath(os.path.dirname(input_source_files_list))
            with open(input_source_files_list, "r") as f:
                for s in f:
                    self.input_data_files.append(s.replace('\n', '').replace(" ", ""))

        self.dc = None
        if input_source_files_list[-6:] == ".djcdc" and not training_data_collection[-6:] == ".djcdc":
            self.dc = DataCollection(input_source_files_list)
        else:
            self.dc = DataCollection(training_data_collection)

        if inputdir is not None:
            self.inputdir = inputdir

        self.model_path = model_path
        if max_files > 0:
            self.input_data_files = self.input_data_files[0:min(max_files, len(self.input_data_files))]
        

    def predict(self, model=None, model_path=None, output_to_file=True):
        if model_path==None:
            model_path = self.model_path

        if model is None:
            if not os.path.exists(model_path):
                raise FileNotFoundError('Model file not found')

        assert model_path is not None or model is not None

        outputs = []
        if output_to_file:
            os.system('mkdir -p ' + self.predict_dir)

        if model is None:
            model = load_model(model_path)

        all_data = []
        for inputfile in self.input_data_files:

            use_inputdir = self.inputdir
            if inputfile[0] == "/":
                use_inputdir = ""
            outfilename = "pred_" + os.path.basename(inputfile)
            
            print('predicting ', use_inputdir +'/' + inputfile)

            td = self.dc.dataclass()

            #also allows for inheriting classes now, like with tracks or special PU
            if not isinstance(td, TrainData_NanoML)  and type(td) is not TrainData_TrackML:
                print(td.__class__.__name__, "not yet fully supported")
            if isinstance(td, TrainData_PreselectionNanoML):
                print(td.__class__.__name__, "support still experimental")

            if inputfile[-5:] == 'djctd':
                if self.unbuffered:
                    td.readFromFile(use_inputdir + "/" + inputfile)
                else:
                    td.readFromFileBuffered(use_inputdir + "/" + inputfile)
            else:
                print('converting ' + inputfile)
                td.readFromSourceFile(use_inputdir + "/" + inputfile, self.dc.weighterobjects, istraining=False)

            gen = TrainDataGenerator()
            # the batch size must be one otherwise we need to play tricks with the row splits later on
            gen.setBatchSize(1)
            gen.setSquaredElementsLimit(False)
            gen.setSkipTooLargeBatches(False)
            gen.setBuffer(td)

            num_steps = gen.getNBatches()
            generator = gen.feedNumpyData()

            dumping_data = []

            thistime = time.time()
            for _ in range(num_steps):
                data_in = next(generator)
                predictions_dict = model(data_in[0])
                for k in predictions_dict.keys():
                    predictions_dict[k] = predictions_dict[k].numpy()
                features_dict = td.createFeatureDict(data_in[0])
                truth_dict = td.createTruthDict(data_in[0])
                
                dumping_data.append([features_dict, truth_dict, predictions_dict])
                
            totaltime = time.time() - thistime
            print('took approx',totaltime/num_steps,'s per endcap (also includes dict building)')

            td.clear()
            gen.clear()
            outfilename = os.path.splitext(outfilename)[0] + '.bin.gz'
            if output_to_file:
                td.writeOutPredictionDict(dumping_data, self.predict_dir + "/" + outfilename)
            outputs.append(outfilename)
            if not output_to_file:
                all_data.append(dumping_data)

        if output_to_file:
            with open(self.predict_dir + "/outfiles.txt", "w") as f:
                for l in outputs:
                    f.write(l + '\n')
        if not output_to_file:
            return all_data


    def predict_all(self, model=None, model_path=None, output_to_file=True):
        """"
        Similar to `predict`, but saves more information
        TODO: Decide on which one we want to use in the future to avoid duplicated code
        """
        if model_path==None:
            model_path = self.model_path

        if model is None:
            if not os.path.exists(model_path):
                raise FileNotFoundError('Model file not found')

        assert model_path is not None or model is not None

        outputs = []
        if output_to_file:
            os.system('mkdir -p ' + self.predict_dir)

        if model is None:
            model = load_model(model_path)

        all_data = []
        eventID = 0
        for inputfile in self.input_data_files:

            use_inputdir = self.inputdir
            if inputfile[0] == "/":
                use_inputdir = ""
            outfilename = "full_pred_" + os.path.basename(inputfile)
            
            print('predicting ', use_inputdir +'/' + inputfile)

            td = self.dc.dataclass()

            #also allows for inheriting classes now, like with tracks or special PU
            if not isinstance(td, TrainData_NanoML)  and type(td) is not TrainData_TrackML:
                print(td.__class__.__name__, "not yet fully supported")
            if isinstance(td, TrainData_PreselectionNanoML):
                print(td.__class__.__name__, "support still experimental")

            if inputfile[-5:] == 'djctd':
                if self.unbuffered:
                    td.readFromFile(use_inputdir + "/" + inputfile)
                else:
                    td.readFromFileBuffered(use_inputdir + "/" + inputfile)
            else:
                print('converting ' + inputfile)
                td.readFromSourceFile(use_inputdir + "/" + inputfile, self.dc.weighterobjects, istraining=False)

            gen = TrainDataGenerator()
            # the batch size must be one otherwise we need to play tricks with the row splits later on
            gen.setBatchSize(1)
            gen.setSquaredElementsLimit(False)
            gen.setSkipTooLargeBatches(False)
            gen.setBuffer(td)

            num_steps = gen.getNBatches()
            generator = gen.feedNumpyData()

            dumping_data = []

            #TODO: This configuration should probably be passed in the __init__ function
            configuration = {
                'beta_threshold' : 0.1,
                'distance_threshold' : 0.5,
                'iou_threshold' : 0.1,
                'matching_mode' : 'iou_max',
                'local_distance_scaling' : True,
                'de_e_cut' : -1,
                'angle_cut' : -1,
            }

            thistime = time.time()
            for _ in range(num_steps):
                data_in = next(generator)
                predictions_dict = model(data_in[0])
                for k in predictions_dict.keys():
                    predictions_dict[k] = predictions_dict[k].numpy()
                features_dict = td.createFeatureDict(data_in[0])
                truth_dict = td.createTruthDict(data_in[0])

                hits2showers = OCHits2ShowersLayer(
                    configuration['beta_threshold'], 
                    configuration['distance_threshold'], 
                    configuration['local_distance_scaling'])
                energy_gatherer = OCGatherEnergyCorrFac()
                processed_pred_dict, pred_shower_alpha_idx = process_endcap(
                    hits2showers, energy_gatherer, features_dict, predictions_dict)
                full_df = pd.DataFrame() 
                for d in [processed_pred_dict, features_dict, truth_dict]:
                    full_df = add_dict_to_df(full_df, d)
                full_df['eventID'] = eventID

                showers_matcher = ShowersMatcher(
                    configuration['matching_mode'],
                    configuration['iou_threshold'], 
                    configuration['de_e_cut'], 
                    configuration['angle_cut'])
                showers_matcher.set_inputs(
                    features_dict=features_dict,
                    truth_dict=truth_dict,
                    predictions_dict=processed_pred_dict,
                    pred_alpha_idx=pred_shower_alpha_idx
                )
                showers_matcher.process()
                showers_df = showers_matcher.get_result_as_dataframe()
                showers_df['eventID'] = eventID
                
                dumping_data.append([full_df, showers_df, pred_shower_alpha_idx])
                
            totaltime = time.time() - thistime
            print('took approx', totaltime/num_steps, \
                's per endcap (also includes transformation into dataframe)')

            td.clear()
            gen.clear()
            outfilename = os.path.splitext(outfilename)[0] + '.bin.gz'
            if output_to_file:
                td.writeOutPredictionDict(dumping_data, self.predict_dir + "/" + outfilename)
            outputs.append(outfilename)
            if not output_to_file:
                all_data.append(dumping_data)

        if output_to_file:
            with open(self.predict_dir + "/outfiles.txt", "w") as f:
                for l in outputs:
                    f.write(l + '\n')

        if not output_to_file:
            return all_data
