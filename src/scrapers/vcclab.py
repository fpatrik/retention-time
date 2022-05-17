import requests
import pandas as pd

from constants import Columns, Datasets

class VcclabScraper:
    """
    Downloads logP data from the ALOGPS 2.1 model at "http://www.vcclab.org/web/alogps"
    """
    def __init__(self):
        pass

    def add_smiles_to_input(self, smiles_list):
        current_smiles = set(self.read_input())
        new_smiles = set(smiles_list)
        self.save_input(
            list(current_smiles | new_smiles)
        )

    def remove_smiles_from_input(self, smiles_list):
        current_smiles = set(self.read_input())
        removed_smiles = set(smiles_list)
        self.save_input(
            list(current_smiles - removed_smiles)
        )

    def clear_input(self):
        self.save_input([])
    
    def read_input(self):
        with open(Datasets.VCCLAB_INPUT) as file:
            return file.read().splitlines()

    def save_input(self, smiles_list):
        with open(Datasets.VCCLAB_INPUT, 'w') as outfile:
            if '' in smiles_list:
                smiles_list.remove('')
            outfile.write('\n'.join(smiles_list))

    def scrape_logp_values(self):
        response = self._send_request()
        result = self._parse_resonse(response)
        self._store_result(result)

    def _send_request(self):
        url = 'http://www.vcclab.org/web/alogps/calc'
    
        files = {'DATAFILE': open(Datasets.VCCLAB_INPUT,'rb')}
        data = {'DATATYPE': 'SMILES'}
        return requests.post(url, files=files, data=data)

    def _parse_resonse(self, response):
        result = []
        for line in response.text.split('<br>')[1: -1]:
            parts = line.split(' ')
            smiles = parts[-1]
            log_p = parts[-3]

            result.append({Columns.SMILES: smiles, Columns.LOGP: log_p})
        
        return pd.DataFrame(result)

    def _store_result(self, result):
        result.to_csv(Datasets.VCCLAB_LOGP, index=False)


if __name__ == '__main__':
    vs = VcclabScraper()
    vs.remove_smiles_from_input(['C1=CC=C(C=C1)CCC(=O)O'])
    vs.add_smiles_to_input(['S(=O)(=O)(CO)O', 'CC(CN)O'])
    vs.add_smiles_to_input(['S(=O)(=O)(CO)O', 'CC(CN)O'])

    vs.scrape_logp_values()