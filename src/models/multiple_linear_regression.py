
import numpy as np
import pandas as pd
import statsmodels.api as sm

from rdkit import Chem
from constants import Datasets
from structures.dataset import Dataset
from structures.feature import create_atom_count_features, create_default_fragment_counts_features

class MultipleLinearRegressionModel():
    def __init__(self):
        self._training_set_molecules = None
        self._training_set_logps = None
        self._test_set_molecules = None
        self._test_set_logps = None

        self._model = None
        self._model_results = None
        self._features = []
    
    def use_dataset(self, X_train, X_test, y_train, y_test):
        self._training_set_molecules, self._test_set_molecules, self._training_set_logps, self._test_set_logps = X_train, X_test, y_train, y_test

    def fit_model(self, print_summary=True):
        calculated_features = self._calculate_features([Chem.MolFromSmiles(smiles) for smiles in self._training_set_molecules])

        Y = pd.DataFrame(self._training_set_logps)
        X = pd.DataFrame(calculated_features, columns=[feature.name for feature in self._features])
        X = sm.add_constant(X)
    
        self._model =  sm.OLS(Y, X)
        self._model_results = self._model.fit()

        if print_summary:
            print(self._model_results.summary())

    def set_features(self, features):
        self._features = features
    
    def predict(self, smiles_list):
        calculated_features = self._calculate_features([Chem.MolFromSmiles(smiles) for smiles in smiles_list])

        X = pd.DataFrame(calculated_features, columns=[feature.name for feature in self._features])
        X = sm.add_constant(X, has_constant='add')

        return self._model_results.get_prediction(X).summary_frame()['mean'].to_list()
        
    def compute_rmse(self, test=True):
        molecules = self._test_set_molecules if test else self._training_set_molecules

        logp_predictions = self.predict(molecules)
        logp_actual = self._test_set_logps if test else self._training_set_logps

        return np.sqrt(np.mean(np.square(
            list(logp_predictions[i] - logp_actual[i][0] for i in range(len(logp_predictions)))
        )))

    def _calculate_features(self, molecules_list):
        return [[feature.compute_for(molecule) for feature in self._features] for molecule in molecules_list]

if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_data(Datasets.OPERA_LOGP)

    mlrm = MultipleLinearRegressionModel()
    atom_count_features = create_atom_count_features(atoms=['C', 'O', 'N', 'F', 'Br', 'Cl'])
    default_fragment_counts_features = create_default_fragment_counts_features()
    mlrm.set_features(atom_count_features + default_fragment_counts_features)

    for X_train, X_val, y_train, y_val in dataset.get_k_fold_train_val_split():
        mlrm.use_dataset(X_train, X_val, y_train, y_val)
        mlrm.fit_model()

        print(mlrm.compute_rmse(test=False))
        print(mlrm.compute_rmse())