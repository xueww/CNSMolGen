import numpy as np
import pandas as pd
import os
import configparser
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
from cnsmolgen import CNSMolGen
from one_hot_encoder import SMILESEncoder
from cnsmolgen_helper import clean_molecule, check_model, check_molecules

np.random.seed(1)

class Trainer():
    def __init__(self, experiment_name='CNSMolGen'):
        self._encoder = SMILESEncoder()
        self._config = configparser.ConfigParser()
        self._config.read(f'../experiments/{experiment_name}.ini')

        # Simplify the configuration parameters by directly accessing the required sections.
        self._model_type = 'CNSMolGen'
        self._experiment_name = experiment_name
        self._hidden_units = int(self._config['MODEL']['hidden_units'])
        self._file_name = '../data/' + self._config['DATA']['data']
        self._encoding_size = int(self._config['DATA']['encoding_size'])
        self._molecular_size = int(self._config['DATA']['molecular_size'])
        self._epochs = int(self._config['TRAINING']['epochs'])
        self._n_folds = int(self._config['TRAINING']['n_folds'])
        self._learning_rate = float(self._config['TRAINING']['learning_rate'])
        self._batch_size = int(self._config['TRAINING']['batch_size'])
        self._samples = int(self._config['EVALUATION']['samples'])
        self._T = float(self._config['EVALUATION']['temp'])
        self._starting_token = self._encoder.encode([self._config['EVALUATION']['starting_token']])
        self._data = self._encoder.encode_from_file(self._file_name)
        # Instantiate the CNSMolGen model directly.

        self._model = CNSMolGen(self._molecular_size, self._encoding_size, self._learning_rate, self._hidden_units)

    def _create_directories(self, base_path):
        """Create directories for storing models, molecules, and statistics."""
        dirs = ['models', 'molecules', 'statistic', 'validation']
        for dir_name in dirs:
            dir_path = os.path.join(base_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)

    def cross_validation(self, stor_dir='../evaluation/', restart=False):
        """Perform cross-validation and store data."""
        base_path = os.path.join(stor_dir, self._experiment_name)
        self._create_directories(base_path)

        kf = KFold(n_splits=self._n_folds, shuffle=True, random_state=2)
        fold = 0
        label = np.argmax(self._data, axis=-1).astype(int)

        for train_index, test_index in kf.split(self._data):
            fold += 1
            train_data, test_data = self._data[train_index], self._data[test_index]
            train_label, test_label = label[train_index], label[test_index]
            train_data, train_label = shuffle(train_data, train_label)

            for epoch in range(self._epochs):
                print(f'Fold: {fold}, Epoch: {epoch}')
                if restart and check_model(self._model_type, self._experiment_name, stor_dir, fold, epoch):
                    # Load existing model and continue training
                    self._model.load(os.path.join(base_path, f'models/model_fold_{fold}_epoch_{epoch}.dat'))
                else:
                    # Perform training
                    self._model.train(train_data, train_label, epochs=1, batch_size=self._batch_size)
                    # Save the model after each epoch
                    self._model.save(os.path.join(base_path, f'models/model_fold_{fold}_epoch_{epoch}.dat'))

                # Validate and store results
                validation_loss = self._model.validate(test_data, test_label)
                with open(os.path.join(base_path, f'validation/validation_fold_{fold}.csv'), 'a') as val_file:
                    val_file.write(f'{epoch},{validation_loss}\n')

                # Sample and store new molecules
                new_molecules = [clean_molecule(self._encoder.decode(self._model.sample(self._starting_token, self._T))[0], self._model_type)
                                 for _ in range(self._samples)]
                pd.DataFrame(new_molecules).to_csv(os.path.join(base_path, f'molecules/molecules_fold_{fold}_epoch_{epoch}.csv'), header=None)

# Example usage:
trainer = Trainer(experiment_name='CNSMolGen')
trainer.cross_validation()