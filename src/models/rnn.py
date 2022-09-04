import numpy as np
from sklearn.model_selection import train_test_split
from constants import Datasets
import tensorflow as tf

from rdkit import Chem

from structures.dataset import Dataset

class RNN:
    def __init__(self, buffer_size=1000, batch_size=64):
        self._training_set_smiles = []
        self._training_set_logps = []
        self._test_set_smiles = []
        self._test_set_logps = []

        self._buffer_size = buffer_size
        self._batch_size = batch_size

        self._model = None

    def use_dataset(self, X_train, X_test, y_train, y_test):
        self._training_set_smiles, self._test_set_smiles, self._training_set_logps, self._test_set_logps = X_train, X_test, y_train, y_test
    
    def load_model(self):
        self._model = tf.keras.models.load_model('./trained_rnn_model')

    def fit_model(self, epochs=308*5):
        x_train, x_validate, y_train, y_validate = train_test_split(self._training_set_smiles, self._training_set_logps, test_size=0.2, random_state=42)

        x_train = [self.preprocess_smiles(smiles) for smiles in x_train]
        x_validate = [self.preprocess_smiles(smiles) for smiles in x_validate]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(self._buffer_size).batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

        validate_dataset = tf.data.Dataset.from_tensor_slices((x_validate, y_validate))
        validate_dataset = validate_dataset.batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

        if self._model is None:
            encoder = tf.keras.layers.TextVectorization(
                standardize=None,
                split='whitespace')
            encoder.adapt(train_dataset.map(lambda text, label: text))

            vocab_size = len(encoder.get_vocabulary())

            self._model = tf.keras.Sequential([
                encoder,
                tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=64,
                    mask_zero=True),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

            self._model.compile(
                loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['mean_squared_error']
            )

        self._model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validate_dataset,
            validation_steps=30
        )

        self._model.save('./trained_rnn_model')

    def predict(self, smiles_list):
        dataset = tf.data.Dataset.from_tensor_slices([self.preprocess_smiles(smiles) for smiles in smiles_list])
        dataset = dataset.batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

        return tf.reshape(self._model.predict(dataset), [-1]).numpy()

    def compute_rmse(self, test=True):
        smiles_list = self._test_set_smiles if test else self._training_set_smiles
        logps_list = self._test_set_logps if test else self._training_set_logps

        dataset = tf.data.Dataset.from_tensor_slices(([self.preprocess_smiles(smiles) for smiles in smiles_list], [item for sublist in logps_list for item in sublist]))
        dataset = dataset.batch(self._batch_size).prefetch(tf.data.AUTOTUNE)
    
        loss, acc = self._model.evaluate(dataset)
        return np.sqrt(loss)

    @staticmethod
    def preprocess_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        encoded_atoms = []
        for atom in mol.GetAtoms():
            atom_index = atom.GetIdx()
            atom_number = atom.GetAtomicNum()
            neighbors = []
            for neighbor in atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx() 
                neighbor_number = neighbor.GetAtomicNum()
                bond_type = mol.GetBondBetweenAtoms(atom_index, neighbor_index).GetBondType()

                neighbors.append(f'{neighbor_number}-{bond_type}')

            encoded_neighbors = ';'.join(sorted(neighbors))
            encoded_atoms.append(f'{atom_number}:{encoded_neighbors}')
        
        return ' '.join(encoded_atoms)

if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_data(Datasets.OPERA_LOGP)

    for X_train, X_val, y_train, y_val in dataset.get_k_fold_train_val_split():
        rnn = RNN()
        rnn.use_dataset(X_train, X_val, y_train, y_val)
        rnn.fit_model()

        print('Test RMSE: ', rnn.compute_rmse())
