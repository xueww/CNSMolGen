"""
Implementation of preprocessing steps for CNSMolGen model.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
import os

np.random.seed(1)

class Preprocessor:

    def __init__(self, name):

        # where name is the name of the file 
        
        # Check if file exists and load data
        self._data = None
        if os.path.isfile(name + '.csv'):
            self._data = pd.read_csv(name + '.csv', header=None).values[:, 0]
        elif os.path.isfile(name + '.tar.xz'):
            self._data = pd.read_csv(name + '.tar.xz', compression='xz', header=None).values[1:-1, 0]
        self._data = np.squeeze(self._data)

    def preprocess(self, aug=1, length=74):
        """
        Preprocess data for CNSMolGen model.
        :param aug:     Data augmentation
        :param length:  Desired length of SMILES strings
        """
        self.add_middle('G')
        self.add_ending('E')
        self.add_sentinel('E')
        self.padding_left_right('A', l=length+3)
        if aug > 1:
            self.add_token_random_padding(start_token='G', pad_token='A', aug=aug, l=3+length*2)

    # Other methods remain unchanged, only removed methods that are not used for  CNSMolGen

    def add_middle(self, token='G'):
        """
        Add token in the middle of each SMILES
        :param token:  token to insert
        """
        for i, s in enumerate(self._data):
            mid = len(s) // 2
            self._data[i] = s[:mid] + token + s[mid:]

    def add_token_random_padding(self, start_token='G', pad_token='A', aug=5, l=0):
        """
        Add start_token at n different random positions and pad to have start_token in the middle of the obtained sequence.
        Method should be applied after add_ending.
        :param start_token: token introduced in the string
        :param pad_token:   token used for padding
        :param aug:         number for data augmentation
        :param l:           length of the final string (if l=0 use length of longest string)
        """
        if l == 0:
            max_l = len(max(self._data, key=len)) - 1
        else:
            max_l = l // 2
        aug_data = []
        for s in self._data:
            l = len(s)
            r = np.random.choice(np.arange(l - 1) + 1, aug, replace=False)
            for r_j in r:
                aug_data.append(s[:r_j].rjust(max_l, pad_token) + start_token + s[r_j:].ljust(max_l, pad_token))
        self._data = np.array(aug_data)

    # Other methods remain unchanged, only removed methods that are not used for  CNSMolGen

    def save_data(self, name='data.csv'):
        """
        Save preprocessed data to a CSV file.
        :param name: Filename for the CSV
        """
        pd.DataFrame(self._data).to_csv(name, header=None, index=None)

    def get_data(self):
        """
        Get the preprocessed data.
        :return: Preprocessed data
        """
        return self._data