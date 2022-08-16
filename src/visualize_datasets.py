import matplotlib.pyplot as plt

from constants import Datasets
from structures.dataset import Dataset
from rdkit import Chem
from chemplot import Plotter
from rdkit.Chem.Descriptors import MolWt

PLOTS_DIR = 'plots'

def get_dataset_size(dataset):
    return len(dataset.smiles_list)

def get_mass_range(dataset):
    min_mass = min(MolWt(Chem.AddHs(Chem.MolFromSmiles(smiles))) for smiles in dataset.smiles_list)
    max_mass = max(MolWt(Chem.AddHs(Chem.MolFromSmiles(smiles))) for smiles in dataset.smiles_list)

    return f'{min_mass} to {max_mass}'

def get_logp_range(dataset):
    min_mass = min(v[0] for v in dataset.logp_list)
    max_mass = max(v[0] for v in dataset.logp_list)

    return f'{min_mass} to {max_mass}'

def plot_logp_distribution(dataset, dataset_name, **kwargs):
    plt.hist([v[0] for v in dataset.logp_list], **kwargs)
    plt.grid()
    plt.ylabel('Count')
    plt.xlabel('LogP')
    plt.title(f'Distribution of LogP for Dataset {dataset_name}')
    plt.savefig(f'{PLOTS_DIR}/logp_distribution_{dataset_name}.png', dpi=300)
    plt.clf()

def plot_molecular_mass_distribution(dataset, dataset_name, **kwargs):
    molecular_masses = [MolWt(Chem.AddHs(Chem.MolFromSmiles(smiles))) for smiles in dataset.smiles_list]
    plt.hist(molecular_masses, **kwargs)
    plt.grid()
    plt.ylabel('Count')
    plt.xlabel('Molecular Mass (Daltons)')
    plt.title(f'Distribution of Molecular Mass for Dataset {dataset_name}')
    plt.savefig(f'{PLOTS_DIR}/molecular_mass_distribution_{dataset_name}.png', dpi=300)
    plt.clf()

def plot_dataset_clusters(datasets, dataset_names):
    smiles = []
    target = []

    for i, dataset in enumerate(datasets):
        smiles += dataset.smiles_list
        target += [dataset_names[i]] * len(dataset.smiles_list)

    cp = Plotter.from_smiles(smiles, target=target, target_type="C")
    cp.tsne()
    cp.visualize_plot(filename=f'{PLOTS_DIR}/dataset_clusters.png')

if __name__ == '__main__':
    datasets = []
    dataset_names = []
    for dataset in [
        (Datasets.OPERA_LOGP, 'Opera'),
        (Datasets.MARTEL_LOGP, 'Martel'),
        (Datasets.SAMPL7_LOGP, 'SAMPL7')
    ]:
        d = Dataset()
        d.load_data(dataset[0])
        datasets.append(d)
        dataset_names.append(dataset[1])
    
    max_logp = max(max(v[0] for v in d.logp_list) for d in datasets)
    min_logp = min(min(v[0] for v in d.logp_list) for d in datasets)
    max_molecular_size = max(max([MolWt(Chem.AddHs(Chem.MolFromSmiles(smiles))) for smiles in d.smiles_list]) for d in datasets)

    for i, dataset in enumerate(datasets):
        dataset_name = dataset_names[i]

        print(f'Size of {dataset_name}: {get_dataset_size(dataset)}')
        print(f'Mass range of {dataset_name}: {get_mass_range(dataset)}')
        print(f'LogP range of {dataset_name}: {get_logp_range(dataset)}')
        plot_logp_distribution(dataset, dataset_name, density=False, range=(min_logp, max_logp), bins=20)
        plot_molecular_mass_distribution(dataset, dataset_name, density=False, range=(0, max_molecular_size), bins=20, color='green')

    #plot_dataset_clusters(datasets, dataset_names)

