"""
Implementation of the sampler to generate SMILES from a trained CNSMolGen model.
"""
import pandas as pd
import numpy as np
import configparser
from cnsmolgen import CNSMolGen
from one_hot_encoder import SMILESEncoder
import os
from helper import clean_molecule, check_valid
np.random.seed(1)
class Sampler():
    def __init__(self, experiment_name):
        self._encoder = SMILESEncoder()
        # Read parameter used during training
        self._config = configparser.ConfigParser()
        self._config.read('../experiments/' + experiment_name + '.ini')
        self._experiment_name = experiment_name
        self._hidden_units = int(self._config['MODEL']['hidden_units'])
        self._file_name = self._config['DATA']['data']
        self._encoding_size = int(self._config['DATA']['encoding_size'])
        self._molecular_size = int(self._config['DATA']['molecular_size'])
        self._learning_rate = float(self._config['TRAINING']['learning_rate'])
        # Initialize CNSMolGen model
        self._model = CNSMolGen(self._molecular_size, self._encoding_size, self._learning_rate, self._hidden_units)
        # Read data
        data_path = '../data/' + self._file_name
        if os.path.isfile(data_path + '.csv'):
            self._data = pd.read_csv(data_path + '.csv', header=None).values[:, 0]
        elif os.path.isfile(data_path + '.tar.xz'):
            self._data = pd.read_csv(data_path + '.tar.xz', compression='xz', header=None).values[1:-1, 0]
        # Clean data from start, end and padding token
        self._data = [clean_molecule(mol_dat, 'CNSMolGen') for mol_dat in self._data]
    def sample(self, N=100, stor_dir='../evaluation', T=0.7, epoch=[9], valid=True, novel=True, unique=True, write_csv=True):
        '''Sample from the CNSMolGen model and store the generated SMILES.
        :param stor_dir: directory where the generated SMILES are saved
        :param N: number of samples
        :param T: Temperature for sampling
        :param epoch: Epochs to use for sampling
        :param valid: If True, only accept valid SMILES
        :param novel: If True, only accept novel SMILES
        :param unique: If True, only accept unique SMILES
        :param write_csv: If True, the generated SMILES are written to CSV
        :return: res_molecules: list with all the generated SMILES
        '''
        res_molecules = []
        print('Sampling: started')
        for e in epoch:
            model_path = f'{stor_dir}/{self._experiment_name}/models/model_epochs_{e}'
            self._model.build(model_path)
            new_molecules = []
            while len(new_molecules) < N:
                new_mol = self._encoder.decode(self._model.sample(self._starting_token, T))[0]
                new_mol = clean_molecule(new_mol, 'CNSMolGen')
                if valid and not check_valid(new_mol):
                    continue
                if unique and new_mol in new_molecules:
                    continue
                if novel and new_mol in self._data:
                    continue
                new_molecules.append(new_mol)
            name = f'molecules_epochs_{e}_T_{T}_N_{N}.csv'
            if unique:
                name = 'unique_' + name
            if valid:
                name = 'valid_' + name
            if novel:
                name = 'novel_' + name
            if write_csv:
                molecules_dir = f'{stor_dir}/{self._experiment_name}/molecules/'
                os.makedirs(molecules_dir, exist_ok=True)
                pd.DataFrame(new_molecules).to_csv(molecules_dir + name, header=None)
            res_molecules.append(new_molecules)
        print('Sampling: done')
        return res_molecules
# Example usage:
# sampler = Sampler('experiment_name')
# molecules = sampler.sample(N=100, T=0.7, epoch=[9])