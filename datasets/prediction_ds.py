import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

import json


class PredictionDataset(Dataset):
    def __init__(self, ds_path: str, evaluation: bool):
        self.ds_path = ds_path
        self.evaluation = evaluation

        # convert the .jsonl file to a list of dicts
        with open(ds_path, 'r') as f:
            data = f.readlines()

        self.data = [json.loads(x.strip()) for x in data]
        self.actual_dataset_length = len(self.data)

        if evaluation:
            self.effective_dataset_length = int(len(self.data) / 10)
        else:
            self.effective_dataset_length = len(self.data)

    def __len__(self) -> int:
        return self.effective_dataset_length

    def __getitem__(self, index: int) -> T_co:
        if not self.evaluation:
            document_id = index
        else:
            document_id = np.random.randint(index * 10, (index + 1) * 10)

        document = self.data[document_id]
        all_nps_spans = [(np['first_token'], np['last_token']) for np in document['nps'].values()]
        document['all_nps_spans'] = torch.tensor(all_nps_spans, dtype=torch.int8)
        return document
