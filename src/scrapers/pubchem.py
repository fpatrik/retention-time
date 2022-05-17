import pandas as pd
import requests
import re
import time

from alive_progress import alive_bar
from constants import Datasets, Columns

class PubChemScraper:
    """
    We have a list of molecules with experimental logP values in the PUBCHEM_INPUT dataset.
    We can scrape them via their API (https://pubchemdocs.ncbi.nlm.nih.gov/pug-view)
    """
    def __init__(self, n_retries=3, requests_per_second=5, save_interval=50):
        self._n_retries = n_retries
        self._requests_per_second = requests_per_second
        self._save_interval = save_interval

        self._molecules_already_scraped = self._get_molecules_already_scraped()
        self._molecules_to_be_scraped = self._get_molecules_to_be_scraped()

    def scrape(self):
        n_scraped_molecules = 0
        with alive_bar(
            len(self._molecules_to_be_scraped),
            title = 'Scraping PubChem',
            dual_line=True
        ) as bar:
            for index, row in self._molecules_to_be_scraped.iterrows():
                smiles = row[Columns.SMILES]
                cid = row[Columns.CID]
                bar.text = f'Processing {smiles}'

                logp = self._scrape_cid(cid)

                if logp is not None:
                    self._molecules_already_scraped = pd.concat([
                        self._molecules_already_scraped,
                        pd.DataFrame([{Columns.SMILES: smiles, Columns.LOGP: logp}])
                    ], ignore_index=True)

                if n_scraped_molecules % self._save_interval == 0:
                    print('saving')
                    self._save_molecules_already_scraped()
                
                n_scraped_molecules += 1
                bar()

        self._save_molecules_already_scraped()

    def _scrape_cid(self, cid):
        attempts = 0
        data = None

        while attempts < self._n_retries:
            try:
                data = self._get_data(cid)
                break
            except:
                attempts += 1
        
        if data is None:
            raise RuntimeError(f'Retries exceeded for cid: {cid}')

        try:
            return self._get_logp_from_data(data)
        except:
            return None
              
    
    def _get_data(self, cid):
        # Ensure we do not exceed the maximum requests per second
        time.sleep(1 / self._requests_per_second)
    
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=LogP'
        resp = requests.get(url=url)
        return resp.json()

    def _get_molecules_to_be_scraped(self):
        pubchem_molecules_with_experimental_logp = self._get_pubchem_molecules_with_experimental_logp()
        
        data_with_scraped_logps = pubchem_molecules_with_experimental_logp.merge(
            self._molecules_already_scraped, on=Columns.SMILES, how='left'
        )

        return data_with_scraped_logps[data_with_scraped_logps[Columns.LOGP].isna()][[Columns.CID, Columns.SMILES]]

    def _save_molecules_already_scraped(self):
        self._molecules_already_scraped.to_csv(Datasets.PUBCHEM_LOGP, index=False)
    
    @staticmethod
    def _get_pubchem_molecules_with_experimental_logp():
        return pd.read_csv(Datasets.PUBCHEM_INPUT)

    @staticmethod
    def _get_molecules_already_scraped():
        try:
            return pd.read_csv(Datasets.PUBCHEM_LOGP)
        except:
            return pd.DataFrame(columns=[Columns.SMILES, Columns.LOGP])
    
    @staticmethod
    def _get_logp_from_data(data):
        """
        This is a bit ugly and might not catch every case. Should do for now.
        """
        records = data['Record']['Section'][0]['Section'][0]['Section'][0]['Information']

        for record in records:
            value = record['Value']

            if 'Number' in value:
                return float(value['Number'][0])
            
            if 'StringWithMarkup'  in value:
                string = value['StringWithMarkup'][0]['String']
                return float(re.findall("\d+\.\d+", string)[0])

if __name__ == '__main__':
    PubChemScraper().scrape()