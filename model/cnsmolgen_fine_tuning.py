"""
Implementation of fine-tuning methods for CNSMolGen model.
"""
import numpy as np
import pandas as pd
import configparser
from cnsmolgen import CNSMolGen
from one_hot_encoder import SMILESEncoder
import os
from helper import clean_molecule, check_model, check_molecules
np.random.seed(1)
class FineTuner():
    def __init__(self, experiment_name='CNSMolGen'):
        self._encoder = SMILESEncoder()
        # Read all parameters from the .ini file
        self._config = configparser.ConfigParser()
        self._config.read('../experiments/' + experiment_name + '.ini')
        self._experiment_name = experiment_name
        self._hidden_units = int(self._config['MODEL']['hidden_units'])
        self._file_name = '../data/' + self._config['DATA']['data']
        self._encoding_size = int(self._config['DATA']['encoding_size'])
        self._molecular_size = int(self._config['DATA']['molecular_size'])
        self._epochs = int(self._config['TRAINING']['epochs'])
        self._learning_rate = float(self._config['TRAINING']['learning_rate'])
        self._batch_size = int(self._config['TRAINING']['batch_size'])
        self._samples = int(self._config['EVALUATION']['samples'])
        self._T = float(self._config['EVALUATION']['temp'])
        self._starting_token = self._encoder.encode([self._config['EVALUATION']['starting_token']])
        self._start_model = self._config['FINETUNING']['start_model']
        # Initialize CNSMolGen model
        self._model = CNSMolGen(self._molecular_size, self._encoding_size, self._learning_rate, self._hidden_units)
        self._data = self._encoder.encode_from_file(self._file_name)

    def fine_tuning(self, stor_dir='../evaluation/', restart=False):
        '''Perform fine-tuning for the CNSMolGen model and store statistics.
        :param stor_dir: directory to store data
        :param restart: whether to restart from the last saved epoch
        '''
        # Create directories
        model_dir = os.path.join(stor_dir, self._experiment_name, 'models')
        stat_dir = os.path.join(stor_dir, self._experiment_name, 'statistic')
        mol_dir = os.path.join(stor_dir, self._experiment_name, 'molecules')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(stat_dir, exist_ok=True)
        os.makedirs(mol_dir, exist_ok=True)
        # Compute labels
        label = np.argmax(self._data, axis=-1).astype(int)

        # Build model
        self._model.build(self._start_model)

        # Store total Statistics
        tot_stat = []

        # only one fold for fine-tuning
        fold = 1
        for i in range(self._epochs):
            print('Epoch:', i+1)

            # Check if current epoch is successfully completed else continue with normal training
            if restart and check_model(self._experiment_name, stor_dir, fold, i) and check_molecules(self._experiment_name, stor_dir, fold, i):
                # Load model
                self._model.build(os.path.join(model_dir, f'model_fold_{fold}_epochs_{i}'))
                # Skip this epoch
                continue
            else:
                restart = False

            # Train model (Data reshaped from (N_samples, N_augmentation, molecular_size, encoding_size)
            # to  (all_SMILES, molecular_size, encoding_size))
            statistic = self._model.train(self._data.reshape(-1, self._molecular_size, self._encoding_size), label.reshape(-1, self._molecular_size), epochs=1, batch_size=self._batch_size)
            tot_stat.append(statistic.tolist())

            # Store model
            self._model.save(os.path.join(model_dir, f'model_fold_{fold}_epochs_{i}'))

            # Sample new molecules
            new_molecules = [clean_molecule(self._encoder.decode(self._model.sample(self._starting_token, self._T))[0]) for _ in range(self._samples)]

            # Store new molecules
            pd.DataFrame(new_molecules).to_csv(os.path.join(mol_dir, f'molecule_fold_{fold}_epochs_{i}.csv'), header=None)

            # Store statistic
            pd.DataFrame(np.array(tot_stat)).to_csv(os.path.join(stat_dir, f'stat_fold_{fold}.csv'), header=None)