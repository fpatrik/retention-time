

import numpy as np
import pandas as pd
from constants import Columns, Datasets
from models.gnn.gnn_model import GNNModel
from models.multiple_linear_regression import MultipleLinearRegressionModel
from models.random_forrest import RandomForrestModel
from models.rnn import RNN
from scrapers.vcclab import VcclabScraper
from structures.dataset import Dataset
from structures.feature import create_atom_count_features, create_default_fragment_counts_features


def evaluate_mlr_model(dataset):
    mlrm = MultipleLinearRegressionModel()
    atom_count_features = create_atom_count_features(atoms=['C', 'O', 'N', 'F', 'Br', 'Cl'])
    default_fragment_counts_features = create_default_fragment_counts_features()
    mlrm.set_features(atom_count_features + default_fragment_counts_features)

    rmses = []
    for X_train, X_val, y_train, y_val in dataset.get_k_fold_train_val_split():
        mlrm.use_dataset(X_train, X_val, y_train, y_val)
        mlrm.fit_model()

        rmses.append(mlrm.compute_rmse())

    return rmses

def evaluate_rf_model(dataset):
    rf = RandomForrestModel()
    atom_count_features = create_atom_count_features(atoms=['C', 'O', 'N', 'F', 'Br', 'Cl'])
    default_fragment_counts_features = create_default_fragment_counts_features()
    rf.set_features(atom_count_features + default_fragment_counts_features)

    rmses = []
    for X_train, X_val, y_train, y_val in dataset.get_k_fold_train_val_split():
        rf.use_dataset(X_train, X_val, y_train, y_val)
        rf.fit_model()

        rmses.append(rf.compute_rmse())
    
    return rmses

def evaluate_rnn_model(dataset):
    rmses = []
    for X_train, X_val, y_train, y_val in dataset.get_k_fold_train_val_split():
        rnn = RNN()
        rnn.use_dataset(X_train, X_val, y_train, y_val)
        rnn.fit_model()

        rmses.append(rnn.compute_rmse())

    return rmses

def evaluate_gnn_model(dataset):
    rmses = []
    for X_train, X_val, y_train, y_val in dataset.get_k_fold_train_val_split():
        gnn = GNNModel()
        gnn.use_dataset(X_train, X_val, y_train, y_val)

        rmse = gnn.fit_model()
        rmses.append(rmse)

    return rmses

def evaluate_alogps_model(dataset, scrape=False):
    smiles_to_logp = {smiles: dataset.logp_list[i] for i, smiles in enumerate(dataset.smiles_list)}

    if scrape:
        scraper = VcclabScraper()
        scraper.clear_input()
        scraper.add_smiles_to_input(dataset.smiles_list)
        scraper.scrape_logp_values()

    result = pd.read_csv(Datasets.VCCLAB_LOGP)

    se = 0
    count = 0
    for index, row in result.iterrows():
        smiles = row[Columns.SMILES]
        logp_estimate = row[Columns.LOGP]

        if smiles not in smiles_to_logp:
            continue

        logp = smiles_to_logp[smiles]
        se += np.square(logp_estimate - logp)
        count += 1
    
    return [np.sqrt(se / count)]

def evaluate_models():
    dataset = Dataset()
    dataset.load_data(Datasets.OPERA_LOGP)
    
    mlr_mean_rmse = np.mean(evaluate_mlr_model(dataset))
    rf_mean_rmse = np.mean(evaluate_rf_model(dataset))
    rnn_mean_rmse = np.mean(evaluate_rnn_model(dataset))
    gnn_mean_rmse = np.mean(evaluate_gnn_model(dataset))

    print(100 * '-')
    print('Multiple Linear Regression Mean Test RMSE: ', mlr_mean_rmse)
    print('Random Forrest Mean Test RMSE: ', rf_mean_rmse)
    print('RNN Mean Test RMSE: ', rnn_mean_rmse)
    print('GNN Mean Test RMSE: ', gnn_mean_rmse)

if __name__ == '__main__':
    #evaluate_models()
    dataset = Dataset()
    dataset.load_data(Datasets.OPERA_LOGP)
    print(evaluate_alogps_model(dataset))