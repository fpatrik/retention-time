from rdkit import Chem
from sklearn.model_selection import train_test_split, KFold
from constants import Columns, Datasets

import pandas as pd

class Dataset:
    def __init__(self):
        self.smiles_list = None
        self.logp_list = None

        self._smiles_list_train = None
        self._smiles_list_test = None
        self._logp_list_train = None
        self._logp_list_test = None

    def load_data(self, path):
        dataset = pd.read_csv(path)
        self.smiles_list, self.logp_list = self.process_dataset(dataset)
        self._smiles_list_train, self._smiles_list_test, self._logp_list_train, self._logp_list_test = train_test_split(self.smiles_list, self.logp_list, test_size=0.2, random_state=42)
    
    def get_k_fold_train_val_split(self, n_splits=5):
        kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        for train_index, test_index in kf.split(self._smiles_list_train):
            X_train, X_val = [self._smiles_list_train[i] for i in train_index], [self._smiles_list_train[i] for i in test_index]
            y_train, y_val = [self._logp_list_train[i] for i in train_index], [self._logp_list_train[i] for i in test_index]

            yield X_train, X_val, y_train, y_val

    @staticmethod
    def process_dataset(dataset):
        smiles_list = []
        logp_list = []
        for index, row in dataset.iterrows():
            smiles = row[Columns.SMILES]
            logp = row[Columns.LOGP]

            if '.' in smiles:
                continue

            try:
                smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
                logp_list.append(logp)
            except:
                continue
        
        return smiles_list, logp_list

if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_data(Datasets.OPERA_LOGP)

    training_set_molecules, test_set_molecules, training_set_logps, test_set_logps = dataset.get_train_test_split()
    pd.DataFrame([{Columns.SMILES: training_set_molecules[i], Columns.LOGP: training_set_logps[i]} for i in range(len(training_set_logps))]).to_csv('data_training.txt', index=False, sep='\t')
    pd.DataFrame([{Columns.SMILES: test_set_molecules[i], Columns.LOGP: test_set_logps[i]} for i in range(len(test_set_logps))]).to_csv('data_test.txt', index=False, sep='\t')