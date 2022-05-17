import timeit
import numpy as np
import torch
import models.gnn.preprocess as pp

from models.gnn.train import MolecularGraphNeuralNetwork, Tester, Trainer
from constants import Datasets
from structures.dataset import Dataset

class GNNModel:
    def __init__(self) -> None:
        self._training_set_molecules = None
        self._training_set_logps = None
        self._test_set_molecules = None
        self._test_set_logps = None

        self._model = None
        self._state_dict = None
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def use_dataset(self, X_train, X_test, y_train, y_test):
        self._training_set_molecules, self._test_set_molecules, self._training_set_logps, self._test_set_logps = X_train, X_test, y_train, y_test

    def fit_model(self, radius=1, dim=50, layer_hidden=6, layer_output=6,
        batch_train=32, batch_test=32, lr=1e-5, lr_decay=0.99, decay_interval=10, iteration=500):

        print('-'*100)

        print('Preprocessing the dataset.')
        print('Just a moment......')
        (dataset_train, dataset_test,
        N_fingerprints) = pp.create_datasets(
            self._training_set_molecules, 
            self._training_set_logps,
            self._test_set_molecules, 
            self._test_set_logps,
            radius,
            self._device
        )

        print('-'*100)

        print('The preprocess has finished!')
        print('# of training data samples:', len(dataset_train))
        print('# of test data samples:', len(dataset_test))
        print('-'*100)

        print('Creating a model.')
        torch.manual_seed(1234)
        np.random.seed(1234)

        if self._model is None:
            self._model = MolecularGraphNeuralNetwork(
                    N_fingerprints, dim, layer_hidden, layer_output, self._device).to(self._device)

        if self._state_dict is not None:
            self._model.load_state_dict(self._state_dict)
            self._model.eval()

        trainer = Trainer(self._model, batch_train, lr)
        tester = Tester(self._model, batch_test)
        print('# of model parameters:',
            sum([np.prod(p.size()) for p in self._model.parameters()]))
        print('-'*100)
        print('Start training.')
        print('The result is saved in the output directory every epoch!')

        
        start = timeit.default_timer()

        for epoch in range(iteration):

            epoch += 1
            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay
            loss_train = trainer.train(dataset_train)

            prediction_test = tester.test_regressor(dataset_test)

            time = timeit.default_timer() - start

            if epoch == 1:
                minutes = time * iteration / 60
                hours = int(minutes / 60)
                minutes = int(minutes - 60 * hours)
                print('The training will finish in about',
                    hours, 'hours', minutes, 'minutes.')
                print('-'*100)
                print('Epoch\tTime(sec)\tLoss_train\t\tRMSE_test')

            result = '\t'.join(map(str, [epoch, time, loss_train, prediction_test]))

            print(result)

        torch.save(self._model.state_dict(), './trained_gnn_model')

        return prediction_test

    def load_model(self):
        self._state_dict = torch.load('./trained_gnn_model')
        

if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_data(Datasets.OPERA_LOGP)

    gnn = GNNModel()
    gnn.use_dataset(dataset)

    gnn.load_model()
    gnn.fit_model()