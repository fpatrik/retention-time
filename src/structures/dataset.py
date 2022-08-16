from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from sklearn.model_selection import train_test_split, KFold
from constants import Columns, Datasets
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

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

    def get_scaffold_split(self, val_size=0.2):
        mols_with_scaffold = []
        scaffold_counts = {}
        for i, smiles in enumerate(self._smiles_list_train):
            scaffold = MurckoScaffoldSmiles(smiles, includeChirality=False)
    
            if scaffold in scaffold_counts:
                scaffold_counts[scaffold] += 1
            else:
                scaffold_counts[scaffold] = 1

            mols_with_scaffold.append((smiles, self._logp_list_train[i], scaffold))
        
        sorted_mols_with_scaffold = sorted(mols_with_scaffold, key=lambda mol: scaffold_counts[mol[2]])

        X_train, X_val, y_train, y_val = [], [], [], []

        for i, mol_with_scaffold in enumerate(sorted_mols_with_scaffold):
            if i < (1-val_size) * len(sorted_mols_with_scaffold):
                X_train.append(mol_with_scaffold[0])
                y_train.append(mol_with_scaffold[1])
            else:
                X_val.append(mol_with_scaffold[0])
                y_val.append(mol_with_scaffold[1])

        return X_train, X_val, y_train, y_val

    @staticmethod
    def process_dataset(dataset):
        smiles_list = []
        props_list = []
        for index, row in dataset.iterrows():
            smiles = row[Columns.SMILES]
            props = [row[column] for column in dataset.columns if column != Columns.SMILES]

            if '.' in smiles:
                continue

            try:
                smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
                props_list.append(props)
            except:
                continue
        
        return smiles_list, props_list

if __name__ == '__main__':
    pass