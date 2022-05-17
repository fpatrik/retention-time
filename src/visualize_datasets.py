import matplotlib.pyplot as plt

from constants import Datasets
from structures.dataset import Dataset
from rdkit import Chem
from chemplot import Plotter

PLOTS_DIR = 'plots'

def get_dataset_size(dataset):
    return len(dataset.smiles_list)

def plot_logp_distribution(dataset, dataset_name, **kwargs):
    plt.hist(dataset.logp_list, **kwargs)
    plt.grid()
    plt.ylabel('Density')
    plt.xlabel('LogP')
    plt.title(f'Distribution of LogP for Dataset {dataset_name}')
    plt.savefig(f'{PLOTS_DIR}/logp_distribution_{dataset_name}.jpeg', dpi=300)
    plt.clf()

def plot_molecular_size_distribution(dataset, dataset_name, **kwargs):
    molecular_sizes = [len(Chem.AddHs(Chem.MolFromSmiles(smiles)).GetAtoms()) for smiles in dataset.smiles_list]
    plt.hist(molecular_sizes, **kwargs)
    plt.grid()
    plt.ylabel('Density')
    plt.xlabel('Molecular Size')
    plt.title(f'Distribution of Molecular Size for Dataset {dataset_name}')
    plt.savefig(f'{PLOTS_DIR}/molecular_size_distribution_{dataset_name}.jpeg', dpi=300)
    plt.clf()

def plot_dataset_clusters(datasets, dataset_names):
    smiles = []
    target = []

    for i, dataset in enumerate(datasets):
        smiles += dataset.smiles_list
        target += [dataset_names[i]] * len(dataset.smiles_list)

    cp = Plotter.from_smiles(smiles, target=target, target_type="C")
    cp.tsne()
    cp.visualize_plot(filename=f'{PLOTS_DIR}/dataset_clusters.jpeg')

if __name__ == '__main__':
    datasets = []
    dataset_names = []
    for dataset in [
        (Datasets.PUBCHEM_LOGP, 'PubChem'),
        (Datasets.OPERA_LOGP, 'Opera'),
        (Datasets.MARTEL_LOGP, 'Martel'),
        (Datasets.STARLIST_LOGP, 'Starlist')
    ]:
        d = Dataset()
        d.load_data(dataset[0])
        datasets.append(d)
        dataset_names.append(dataset[1])
    
    max_logp = max(max(d.logp_list) for d in datasets)
    min_logp = min(min(d.logp_list) for d in datasets)
    max_molecular_size = max(max([len(Chem.AddHs(Chem.MolFromSmiles(smiles)).GetAtoms()) for smiles in d.smiles_list]) for d in datasets)

    for i, dataset in enumerate(datasets):
        dataset_name = dataset_names[i]

        print(f'Size of {dataset_name}: {get_dataset_size(dataset)}')
        plot_logp_distribution(dataset, dataset_name, density=True, range=[min_logp, max_logp], bins=20)
        plot_molecular_size_distribution(dataset, dataset_name, density=True, range=[0, max_molecular_size], bins=20)

    plot_dataset_clusters(datasets, dataset_names)

