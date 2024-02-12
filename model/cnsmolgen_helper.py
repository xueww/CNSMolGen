from rdkit import Chem
import os

def clean_molecule(mol):
    ''' Clean the SMILES string by removing padding tokens specific to CNSMolGen model.
    :param mol: SMILES string with padding
    :return: Cleaned SMILES string
    '''
    # For CNSMolGen, remove start ('G') and end ('E') padding tokens
    mol = mol.split('G')[1].split('E')[0]
    return mol

def check_valid(mol):
    '''Check if SMILES is valid
    :param mol: SMILES string
    :return: True if valid, False otherwise
    '''
    if mol == '':
        return False
    mol = Chem.MolFromSmiles(mol, sanitize=True)
    return mol is not None

def check_model(model_name, stor_dir, fold, epoch):
    '''Check if the model file exists for a given fold and epoch.
    :param model_name: Name of the model
    :param stor_dir: Directory of stored data
    :param fold: Fold to check
    :param epoch: Epoch to check
    :return: True if model file exists, False otherwise
    '''
    model_path = os.path.join(stor_dir, model_name, 'models', f'model_fold_{fold}_epochs_{epoch}.dat')
    return os.path.isfile(model_path)

def check_molecules(model_name, stor_dir, fold, epoch):
    '''Check if the molecules file exists for a given fold and epoch.
    :param model_name: Name of the model
    :param stor_dir: Directory of stored data
    :param fold: Fold to check
    :param epoch: Epoch to check
    :return: True if molecules file exists, False otherwise
    '''
    molecules_path = os.path.join(stor_dir, model_name, 'molecules', f'molecule_fold_{fold}_epochs_{epoch}.csv')
    return os.path.isfile(molecules_path)