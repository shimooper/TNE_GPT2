from typing import List

from torch import tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

import os.path
import json
import numpy as np
import pandas as pd

PKL_FILE_NAME_SUFFIX = "nps_v2.pkl"

class NextSpanPredictionDataset(Dataset):
    def __init__(self, ds_path: str, prepositions: List[str]):
        self.ds_path = ds_path

        # convert the .jsonl file to a list of dicts
        l = open(self.ds_path,"rb").read().decode('utf-8').split("\n")
        json_dataset = [json.loads(x) for x in l[:-1]]

        self.prepositions = prepositions

        self.data = {'id': [], 'paragraph': [], 'tokens': []}
        for item in json_dataset:
            self.data['paragraph'].append(item['text'])
            self.data['tokens'].append(item['tokens'])
            self.data['id'].append(item['id'])
        self.data = pd.DataFrame(self.data, index=self.data['id'])

        self.relations = {'id': [], 'anchor': [], 'preposition': [], 'complement': []}
        self.anchors = {'id': [], 'np_id': [], 'first_token': [], 'last_token': []}
        self.complements = {'id': [], 'np_id': [], 'first_token': [], 'last_token': []}

        saved = False
        pickle_path = f"{os.path.splitext(ds_path)[0]}_{PKL_FILE_NAME_SUFFIX}"
        try:
            self.nps = pd.read_pickle(pickle_path)
            saved = True
        except:
            self.nps = {'id': [], 'np_id': [], 'first_token': [], 'last_token': [], 'text': [],'is_anchor': []}

            for item in json_dataset:
                for np_id in item['nps']:
                    np = item['nps'][np_id]
                    self.nps['id'].append(item['id'])
                    self.nps['np_id'].append(np['id'])
                    self.nps['first_token'].append(np['first_token'])
                    self.nps['last_token'].append(np['last_token'])
                    np_tokens = self.data.loc[item['id']]['tokens'][np['first_token']:np['last_token']+1]
                    self.nps['text'].append(' '.join(np_tokens))
                    self.nps['is_anchor'].append(False)
            self.nps = pd.DataFrame(self.nps, index=self.nps['id'])

        for i, item in enumerate(json_dataset):
            if not saved:
                print("item #%d / %d" % (i, len(json_dataset)))
            for j, rel in enumerate(item['np_relations']):
                self.relations['anchor'].append(rel['anchor'])
                self.relations['preposition'].append(rel['preposition'])
                self.relations['complement'].append(rel['complement'])
                self.relations['id'].append(item['id'])
                if not saved:
                    self.nps.loc[(self.nps['id'] == item['id']) & (self.nps['np_id'] == rel['anchor']), 'is_anchor'] = True

        if not saved:
            print("Saving pickle...")
            self.nps.to_pickle(pickle_path)
        self.relations = pd.DataFrame(self.relations)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> T_co:
        ds_id = self.data.iloc[index]['id']
        paragraph = self.data.iloc[index]['paragraph']

        curr_paragraph_nps = self.data.iloc[[index]].merge(self.nps, on='id')

        assert len(curr_paragraph_nps[curr_paragraph_nps['is_anchor']]) > 0
        sample_anchor = curr_paragraph_nps[curr_paragraph_nps['is_anchor']].sample()

        relevant_relations = self.relations.loc[(self.relations['id'] == ds_id) &
                                                (self.relations['anchor'] == sample_anchor['np_id'].values[0])]
        relevant_preps = relevant_relations['preposition'].values
        # with prob. 1/2 choose prep from relevant preps, otherwise try to choose it from the others
        if np.random.binomial(1, 0.5):
            prep = np.random.choice(relevant_preps)
        else:
            non_relevant_preps = list(set(self.prepositions) - set(relevant_preps))
            if non_relevant_preps:
                prep = np.random.choice(non_relevant_preps)
            else:
                prep = np.random.choice(relevant_preps)

        relevant_relations = relevant_relations.loc[(relevant_relations['preposition'] == prep)]
        relevant_complements = pd.merge(
            relevant_relations, self.nps,
            left_on=['id', 'complement'], right_on=['id', 'np_id'], how='left')

        span_list = []
        #if not relevant_complements.empty:
        #    print("relevant_complements:", relevant_complements)
        for _, row in relevant_complements.iterrows():
            span_list.append([int(row['first_token']), int(row['last_token'])])

        prompt = f"{paragraph}\n\nQ: {sample_anchor['text'].values[0]} {prep}\nA: "

        text_id = sample_anchor['id'].values[0]
        anchor_id = sample_anchor['np_id'].values[0]
        prep_id = prep
        return text_id, anchor_id, prep_id, prompt, tensor(span_list)
