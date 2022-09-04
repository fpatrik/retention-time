import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch_geometric.nn import PNAConv, GeneralConv
from torch_geometric.seed import seed_everything

from rdkit import Chem
from rdkit import DataStructs
from constants import Columns, Datasets
from structures.dataset import Dataset
from structures.feature import Feature, create_atom_count_features, create_default_fragment_counts_features
from scrapers.vcclab import VcclabScraper
from models.multiple_linear_regression import MultipleLinearRegressionModel
from models.random_forrest import RandomForrestModel
from models.rnn import RNN
from models.pytorch_gnn.pytorch_gnn import GraphNeuralNetModel
from models.pytorch_gnn.pytorch_gnn_multi import MultiTaskGraphNeuralNetModel

PLOTS_DIR = 'plots'

def evaluate_mlr_model(dataset_train, dataset_martel, dataset_sampl7):
    rmses_test = []
    rmses_train = []
    rmses_martel = []
    rmses_sampl7 = []

    atom_count_features = create_atom_count_features(atoms=['C', 'O', 'N', 'F', 'Br', 'Cl'])
    default_fragment_counts_features = create_default_fragment_counts_features()
    
    for i in range(5):
        seed_everything(i)
        mlrm = MultipleLinearRegressionModel()
        mlrm.set_features(atom_count_features + default_fragment_counts_features)
        mlrm.use_dataset(dataset_train._smiles_list_train, dataset_train._smiles_list_test, dataset_train._logp_list_train, dataset_train._logp_list_test)
        mlrm.fit_model()

        rmses_test.append(mlrm.compute_rmse())
        rmses_train.append(mlrm.compute_rmse(test=False))

        mlrm.use_dataset(dataset_train._smiles_list_train, dataset_martel.smiles_list, dataset_train._logp_list_train, dataset_martel.logp_list)
        rmses_martel.append(mlrm.compute_rmse())
        pd.DataFrame([{Columns.LOGP: v} for v in mlrm.predict(dataset_martel.smiles_list)]).to_csv(f'./output/mlr/predictions_{i}.csv', index=False)

        mlrm.use_dataset(dataset_train._smiles_list_train, dataset_sampl7.smiles_list, dataset_train._logp_list_train, dataset_sampl7.logp_list)
        rmses_sampl7.append(mlrm.compute_rmse())

    return rmses_test, rmses_train, rmses_martel, rmses_sampl7

def evaluate_rf_model(dataset_train, dataset_martel, dataset_sampl7):
    rmses_test = []
    rmses_train = []
    rmses_martel = []
    rmses_sampl7 = []

    atom_count_features = create_atom_count_features(atoms=['C', 'O', 'N', 'F', 'Br', 'Cl'])
    default_fragment_counts_features = create_default_fragment_counts_features()

    for i in range(5):
        seed_everything(i)

        rf = RandomForrestModel()
        rf.set_features(atom_count_features + default_fragment_counts_features)

        rf.use_dataset(dataset_train._smiles_list_train, dataset_train._smiles_list_test, dataset_train._logp_list_train, dataset_train._logp_list_test)
        rf.fit_model()

        rmses_test.append(rf.compute_rmse())
        rmses_train.append(rf.compute_rmse(test=False))

        rf.use_dataset(dataset_train._smiles_list_train, dataset_martel.smiles_list, dataset_train._logp_list_train, dataset_martel.logp_list)
        rmses_martel.append(rf.compute_rmse())
        pd.DataFrame([{Columns.LOGP: v} for v in rf.predict(dataset_martel.smiles_list)]).to_csv(f'./output/rf/predictions_{i}.csv', index=False)

        rf.use_dataset(dataset_train._smiles_list_train, dataset_sampl7.smiles_list, dataset_train._logp_list_train, dataset_sampl7.logp_list)
        rmses_sampl7.append(rf.compute_rmse())
    
    return rmses_test, rmses_train, rmses_martel, rmses_sampl7

def evaluate_rnn_model(dataset_train, dataset_martel, dataset_sampl7):
    rmses_test = []
    rmses_train = []
    rmses_martel = []
    rmses_sampl7 = []

    for i in range(5):
        seed_everything(i)

        rnn = RNN()
        rnn.use_dataset(dataset_train._smiles_list_train, dataset_train._smiles_list_test, dataset_train._logp_list_train, dataset_train._logp_list_test)
        rnn.fit_model(epochs=1000)

        rmses_test.append(rnn.compute_rmse())
        rmses_train.append(rnn.compute_rmse(test=False))

        rnn.use_dataset(dataset_train._smiles_list_train, dataset_martel.smiles_list, dataset_train._logp_list_train, dataset_martel.logp_list)
        rmses_martel.append(rnn.compute_rmse())
        pd.DataFrame([{Columns.LOGP: v} for v in rnn.predict(dataset_martel.smiles_list)]).to_csv(f'./output/rnn/predictions_{i}.csv', index=False)

        rnn.use_dataset(dataset_train._smiles_list_train, dataset_sampl7.smiles_list, dataset_train._logp_list_train, dataset_sampl7.logp_list)
        rmses_sampl7.append(rnn.compute_rmse())

    return rmses_test, rmses_train, rmses_martel, rmses_sampl7

def evaluate_gnn_model(dataset_train, dataset_martel, dataset_sampl7):
    rmses_test = []
    rmses_train = []
    rmses_martel = []
    rmses_sampl7 = []

    for i in range(5):
        seed_everything(i)
        gnn = GraphNeuralNetModel()
        gnn.use_dataset(dataset_train._smiles_list_train, dataset_train._smiles_list_test, dataset_train._logp_list_train, dataset_train._logp_list_test)
        gnn.fit_model(epochs=2000, lr=0.0001, lr_decay=0.999)

        rmses_test.append(gnn.compute_rmse())
        rmses_train.append(gnn.compute_rmse(test=False))

        gnn.use_dataset(dataset_train._smiles_list_train, dataset_martel.smiles_list, dataset_train._logp_list_train, dataset_martel.logp_list)
        rmses_martel.append(gnn.compute_rmse())
        pd.DataFrame([{Columns.LOGP: v} for v in gnn.predict(dataset_martel.smiles_list)]).to_csv(f'./output/gnn/predictions_{i}.csv', index=False)

        gnn.use_dataset(dataset_train._smiles_list_train, dataset_sampl7.smiles_list, dataset_train._logp_list_train, dataset_sampl7.logp_list)
        rmses_sampl7.append(gnn.compute_rmse())

    return rmses_test, rmses_train, rmses_martel, rmses_sampl7

def evaluate_gnn_pna_model(dataset_train, dataset_martel, dataset_sampl7):
    rmses_test = []
    rmses_train = []
    rmses_martel = []
    rmses_sampl7 = []

    for i in range(5):
        seed_everything(i)
        gnn = GraphNeuralNetModel({
            'convolutional_layer': PNAConv,
            'convolutional_layer_args': {
                'aggregators': ['sum', 'max', 'var'],
                'scalers': ['identity', 'amplification', 'attenuation'],
                'deg': True
            },
            'dropout': 0.5
        })
        gnn.use_dataset(dataset_train._smiles_list_train, dataset_train._smiles_list_test, dataset_train._logp_list_train, dataset_train._logp_list_test)
        gnn.fit_model(epochs=2000, lr=0.0001, lr_decay=0.999)

        rmses_test.append(gnn.compute_rmse())
        rmses_train.append(gnn.compute_rmse(test=False))

        gnn.use_dataset(dataset_train._smiles_list_train, dataset_martel.smiles_list, dataset_train._logp_list_train, dataset_martel.logp_list)
        rmses_martel.append(gnn.compute_rmse())
        pd.DataFrame([{Columns.LOGP: v} for v in gnn.predict(dataset_martel.smiles_list)]).to_csv(f'./output/gnn_pna/predictions_{i}.csv', index=False)

        gnn.use_dataset(dataset_train._smiles_list_train, dataset_sampl7.smiles_list, dataset_train._logp_list_train, dataset_sampl7.logp_list)
        rmses_sampl7.append(gnn.compute_rmse())

    return rmses_test, rmses_train, rmses_martel, rmses_sampl7

def evaluate_gnn_attention_model(dataset_train, dataset_martel, dataset_sampl7):
    rmses_test = []
    rmses_train = []
    rmses_martel = []
    rmses_sampl7 = []

    for i in range(5):
        seed_everything(i)
        gnn = GraphNeuralNetModel({
            'convolutional_layer': GeneralConv,
            'convolutional_layer_args': {
                'attention': True
            }
        })
        gnn.use_dataset(dataset_train._smiles_list_train, dataset_train._smiles_list_test, dataset_train._logp_list_train, dataset_train._logp_list_test)
        gnn.fit_model(epochs=2000, lr=0.0001, lr_decay=0.999)

        rmses_test.append(gnn.compute_rmse())
        rmses_train.append(gnn.compute_rmse(test=False))

        gnn.use_dataset(dataset_train._smiles_list_train, dataset_martel.smiles_list, dataset_train._logp_list_train, dataset_martel.logp_list)
        rmses_martel.append(gnn.compute_rmse())
        pd.DataFrame([{Columns.LOGP: v} for v in gnn.predict(dataset_martel.smiles_list)]).to_csv(f'./output/gnn_attention/predictions_{i}.csv', index=False)

        gnn.use_dataset(dataset_train._smiles_list_train, dataset_sampl7.smiles_list, dataset_train._logp_list_train, dataset_sampl7.logp_list)
        rmses_sampl7.append(gnn.compute_rmse())

    return rmses_test, rmses_train, rmses_martel, rmses_sampl7

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
    
    return np.mean([np.sqrt(se / count)])

def evaluate_models():
    dataset_train = Dataset()
    dataset_train.load_data(Datasets.OPERA_EX_MARTEL_AND_SAMPL7_LOGP)

    dataset_martel = Dataset()
    dataset_martel.load_data(Datasets.MARTEL_LOGP)

    dataset_sampl7 = Dataset()
    dataset_sampl7.load_data(Datasets.SAMPL7_LOGP)
    
    mlr_rmses_test, mlr_rmses_train, mlr_rmses_martel, mlr_rmses_sample7 = evaluate_mlr_model(dataset_train, dataset_martel, dataset_sampl7)
    rf_rmses_test, rf_rmses_train, rf_rmses_martel, rf_rmses_sample7 = evaluate_rf_model(dataset_train, dataset_martel, dataset_sampl7)
    rnn_rmses_test, rnn_rmses_train, rnn_rmses_martel, rnn_rmses_sample7 = evaluate_rnn_model(dataset_train, dataset_martel, dataset_sampl7)
    gnn_rmses_test, gnn_rmses_train, gnn_rmses_martel, gnn_rmses_sample7 = evaluate_gnn_model(dataset_train, dataset_martel, dataset_sampl7)
    gnn_pna_rmses_test, gnn_pna_rmses_train, gnn_pna_rmses_martel, gnn_pna_rmses_sample7 = evaluate_gnn_pna_model(dataset_train, dataset_martel, dataset_sampl7)
    gnn_attention_rmses_test, gnn_attention_rmses_train, gnn_attention_rmses_martel, gnn_attention_rmses_sample7 = evaluate_gnn_attention_model(dataset_train, dataset_martel, dataset_sampl7)

    print(100 * '-')
    print('Multiple Linear Regression Train RMSE: ', np.mean(mlr_rmses_train), ' +- ', np.std(mlr_rmses_train))
    print('Multiple Linear Regression Test RMSE: ', np.mean(mlr_rmses_test), ' +- ', np.std(mlr_rmses_test))
    print('Multiple Linear Regression Martel RMSE: ', np.mean(mlr_rmses_martel), ' +- ', np.std(mlr_rmses_martel))
    print('Multiple Linear Regression Sampl7 RMSE: ', np.mean(mlr_rmses_sample7), ' +- ', np.std(mlr_rmses_sample7))
    print('Random Forrest Train RMSE: ', np.mean(rf_rmses_train), ' +- ', np.std(rf_rmses_train))
    print('Random Forrest Test RMSE: ', np.mean(rf_rmses_test), ' +- ', np.std(rf_rmses_test))
    print('Random Forrest Martel RMSE: ', np.mean(rf_rmses_martel), ' +- ', np.std(rf_rmses_martel))
    print('Random Forrest Sampl7 RMSE: ', np.mean(rf_rmses_sample7), ' +- ', np.std(rf_rmses_sample7))
    print('RNN Train RMSE: ', np.mean(rnn_rmses_train), ' +- ', np.std(rnn_rmses_train))
    print('RNN Test RMSE: ', np.mean(rnn_rmses_test), ' +- ', np.std(rnn_rmses_test))
    print('RNN Martel RMSE: ', np.mean(rnn_rmses_martel), ' +- ', np.std(rnn_rmses_martel))
    print('RNN Sampl7 RMSE: ', np.mean(rnn_rmses_sample7), ' +- ', np.std(rnn_rmses_sample7))
    print('GNN Train RMSE: ', np.mean(gnn_rmses_train), ' +- ', np.std(gnn_rmses_train))
    print('GNN Test RMSE: ', np.mean(gnn_rmses_test), ' +- ', np.std(gnn_rmses_test))
    print('GNN Martel RMSE: ', np.mean(gnn_rmses_martel), ' +- ', np.std(gnn_rmses_martel))
    print('GNN Sampl7 RMSE: ', np.mean(gnn_rmses_sample7), ' +- ', np.std(gnn_rmses_sample7))
    print('GNN PNA Train RMSE: ', np.mean(gnn_pna_rmses_train), ' +- ', np.std(gnn_pna_rmses_train))
    print('GNN PNA Test RMSE: ', np.mean(gnn_pna_rmses_test), ' +- ', np.std(gnn_pna_rmses_test))
    print('GNN PNA Martel RMSE: ', np.mean(gnn_pna_rmses_martel), ' +- ', np.std(gnn_pna_rmses_martel))
    print('GNN PNA Sampl7 RMSE: ', np.mean(gnn_pna_rmses_sample7), ' +- ', np.std(gnn_pna_rmses_sample7))
    print('GNN Attention Train RMSE: ', np.mean(gnn_attention_rmses_train), ' +- ', np.std(gnn_attention_rmses_train))
    print('GNN Attention Test RMSE: ', np.mean(gnn_attention_rmses_test), ' +- ', np.std(gnn_attention_rmses_test))
    print('GNN Attention Martel RMSE: ', np.mean(gnn_attention_rmses_martel), ' +- ', np.std(gnn_attention_rmses_martel))
    print('GNN Attention Sampl7 RMSE: ', np.mean(gnn_attention_rmses_sample7), ' +- ', np.std(gnn_attention_rmses_sample7))

def evaluate_model_correlation_and_error_distribution():
    dataset_martel = Dataset()
    dataset_martel.load_data(Datasets.MARTEL_LOGP)

    dataset_opera = Dataset()
    dataset_opera.load_data(Datasets.OPERA_EX_MARTEL_AND_SAMPL7_LOGP)

    martel_list = [v[0] for v in dataset_martel.logp_list]
    martel_smiles_list = dataset_martel.smiles_list

    model_predictions = {}
    model_errors = {}
    models = ['mlr', 'rf', 'rnn', 'gnn', 'gnn_pna', 'gnn_attention', 'gnn_multi']
    for model in models:
        for i in range(5):
            result = pd.read_csv(f'./output/{model}/predictions_{i}.csv')[Columns.LOGP].to_list()

            if model not in model_predictions:
                model_predictions[model] = []
                model_errors[model] = []

            model_predictions[model].append(result)
            model_errors[model].append([prediction - martel_list[i] for i, prediction in enumerate(result)])

    print('Correlation matrix:')
    print(np.corrcoef([np.mean(model_errors[model], axis=0) for model in models]))

    mean_predictions = np.mean([np.mean(model_predictions[model], axis=0) for model in models], axis=0)
    print('Mean prediction RMSE:')
    print(np.sqrt(np.mean(np.square([mean_prediction - martel_list[i] for i, mean_prediction in enumerate(mean_predictions)]))))

    error_bins = {}
    for model in models:
        error_bins[model] = {}
        for i in range(5):
            for j, prediction in enumerate(model_predictions[model][i]):
                bin = None
                if np.abs(prediction - martel_list[j]) < 0.5:
                    bin = '0-0.5'
                elif np.abs(prediction - martel_list[j]) < 1:
                    bin = '0.5-1'
                elif np.abs(prediction - martel_list[j]) < 1.5:
                    bin = '1-1.5'
                elif np.abs(prediction - martel_list[j]) < 2:
                    bin = '1.5-2'
                else:
                    bin = '2<'

                if bin not in error_bins[model]:
                    error_bins[model][bin] = 0

                # Such that we get percentages
                error_bins[model][bin] += 100 / (5 * len(martel_list))
    
    print('Error distributions:')
    print(error_bins)

    for model in models:
        mean_predictions = np.mean(model_predictions[model], axis=0)
        model_errors = [mean_predictions[i] - martel_list[i] for i in range(len(mean_predictions))]

        plt.scatter(martel_list, model_errors, c='g', alpha=0.5)
        coef = np.polyfit(martel_list, model_errors,1)
        poly1d_fn = np.poly1d(coef)
        plt.plot([-10, 10], poly1d_fn([-10, 10]), '--k', label=f'y = {poly1d_fn.c[1]:.2f} {poly1d_fn.c[0]:.2f} x')
        plt.title(f'Prediction Error vs Measured logP for {model.upper() if model != "gnn_multi" else "GNN Multitask"} Model')
        plt.xlabel('Measured LogP')
        plt.ylabel('Prediction Error')
        plt.xlim((0, 8))
        plt.ylim((-4, 4))
        plt.legend(fontsize=9)
        plt.savefig(f'{PLOTS_DIR}/martel_errors_{model}.png', dpi=300)
        plt.clf()
    
    for model in models:
        mean_predictions = np.mean(model_predictions[model], axis=0)
        absolute_model_errors = np.abs([mean_predictions[i] - martel_list[i] for i in range(len(mean_predictions))])

        print(model)
        atom_count_features = create_atom_count_features(atoms=['C', 'O', 'N', 'F', 'Br', 'Cl'])
        default_fragment_counts_features = create_default_fragment_counts_features()
        mlr = MultipleLinearRegressionModel()
        mlr.use_dataset(martel_smiles_list, None, absolute_model_errors, None)
        mlr.set_features(atom_count_features)
        mlr.fit_model()

    
def evaluate_gnn_multi():
    dataset = Dataset()
    dataset.load_data(Datasets.OPERA_EX_MARTEL_AND_SAMPL7_LOGP_MULTITASK)

    dataset_logp = Dataset()
    dataset_logp.load_data(Datasets.OPERA_EX_MARTEL_AND_SAMPL7_LOGP)

    dataset_martel = Dataset()
    dataset_martel.load_data(Datasets.MARTEL_LOGP)

    dataset_sampl7 = Dataset()
    dataset_sampl7.load_data(Datasets.SAMPL7_LOGP)

    rmses_test = []
    rmses_train = []
    rmses_martel = []
    rmses_sampl7 = []

    for i in range(5):
        seed_everything(i)
        model = MultiTaskGraphNeuralNetModel()

        model.use_dataset(dataset._smiles_list_train, dataset._smiles_list_test, dataset._logp_list_train, dataset._logp_list_test)
        model.fit_model(epochs=2000, lr=0.0001, mode='both')

        model.use_dataset(dataset_logp._smiles_list_train, dataset_logp._smiles_list_test, dataset_logp._logp_list_train, dataset_logp._logp_list_test)
        rmses_train.append(model.compute_rmse(test=False, mode='logp'))
        rmses_test.append(model.compute_rmse(test=True, mode='logp'))

        model.use_dataset(dataset_logp._smiles_list_train, dataset_martel.smiles_list, dataset_logp._logp_list_train, dataset_martel.logp_list)
        rmses_martel.append(model.compute_rmse(mode='logp'))

        model.use_dataset(dataset_logp._smiles_list_train, dataset_sampl7.smiles_list, dataset_logp._logp_list_train, dataset_sampl7.logp_list)
        rmses_sampl7.append(model.compute_rmse(mode='logp'))

        pd.DataFrame([{Columns.LOGP: v[0].item()} for v in model.predict(dataset_martel.smiles_list, mode='logp')]).to_csv(f'./output/gnn_multi/predictions_{i}.csv', index=False)
        model.save_model(f'./output/gnn_multi/model_{i}/')

    print('GNN Multi Train RMSE: ', np.mean(rmses_train), ' +- ', np.std(rmses_train))
    print('GNN Multi Test RMSE: ', np.mean(rmses_test), ' +- ', np.std(rmses_test))
    print('GNN Multi Martel RMSE: ', np.mean(rmses_martel), ' +- ', np.std(rmses_martel))
    print('GNN Multi Sampl7 RMSE: ', np.mean(rmses_sampl7), ' +- ', np.std(rmses_sampl7))

def evaluate_martel_weighted_training_data_model(sampling='martel'):
    seed_everything(0)
    dataset_logp = Dataset()
    dataset_logp.load_data(Datasets.OPERA_EX_MARTEL_AND_SAMPL7_LOGP)

    dataset_martel = Dataset()
    dataset_martel.load_data(Datasets.MARTEL_LOGP)

    training_smiles = []
    training_logps = []

    bins = []
    for i in range(-2, 8):
        if sampling == 'uniform':
            bins.append((i, i + 1, int(10000 / 11)))
        else:
            bins.append((
                i,
                i + 1,
                int(len([1 for logp in dataset_martel.logp_list if logp[0] > i and logp[0] <= i + 1]) * 10000 / len(dataset_martel.logp_list))
            ))

    for b in bins:
        molecules_added = 0
        while molecules_added < b[2]:
            random_opera_molecule = np.random.randint(0, len(dataset_logp.smiles_list), size=1)[0]
            logp = dataset_logp.logp_list[random_opera_molecule][0]

            if logp > b[0] and logp <= b[1]:
                training_smiles.append(dataset_logp.smiles_list[random_opera_molecule])
                training_logps.append(dataset_logp.logp_list[random_opera_molecule])
                molecules_added += 1

    atom_count_features = create_atom_count_features(atoms=['C', 'O', 'N', 'F', 'Br', 'Cl'])
    default_fragment_counts_features = create_default_fragment_counts_features()
        
    mlrm = MultipleLinearRegressionModel()
    mlrm.set_features(atom_count_features + default_fragment_counts_features)
    mlrm.use_dataset(training_smiles, dataset_martel.smiles_list, training_logps, dataset_martel.logp_list)
    mlrm.fit_model()

    print(mlrm.compute_rmse())


if __name__ == '__main__':
    evaluate_models()
    evaluate_gnn_multi()
    evaluate_model_correlation_and_error_distribution()
    evaluate_martel_weighted_training_data_model(sampling='martel')
    evaluate_martel_weighted_training_data_model(sampling='uniform')