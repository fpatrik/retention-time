from rdkit.Chem import Fragments

class Feature:
    def __init__(self, name, feature_lambda):
        self.name = name
        self._feature_lambda = feature_lambda

    def __eq__(self, other):
        return self.name == other.name

    def compute_for(self, molecule):
        return self._feature_lambda(molecule)

def create_atom_count_features(atoms=[]):
    def create_atom_counter(atom_symbol):
            def get_atom_count(molecule):
                if molecule is None:
                    return 0
                return sum(atom.GetSymbol() == atom_symbol for atom in molecule.GetAtoms())
            return get_atom_count

    return [Feature(f'{atom} count', create_atom_counter(atom)) for atom in atoms]

def create_default_fragment_counts_features():
    rdkit_fragments = [
        method_name for method_name in dir(Fragments) if 
        (
            callable(getattr(Fragments, method_name)) and
            not method_name.startswith('_')
        )
    ]
    def create_fragment_counter(fragment):
        def get_fragment_count(molecule):
            if molecule is None:
                return 0
            return getattr(Fragments, fragment)(molecule)
        return get_fragment_count

    return [Feature(f'{fragment} count', create_fragment_counter(fragment)) for fragment in rdkit_fragments]
