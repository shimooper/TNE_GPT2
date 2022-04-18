from typing import Dict

import torch.cuda
from torch.utils.data import DataLoader
from torch import nn, Tensor
from tqdm import tqdm

import sys
sys.path.append('../..')
sys.path.append('..')
sys.path.append('/home/joberant/NLP_2122/idangrosbard/TNE')
sys.path.append('/home/joberant/NLP_2122/idangrosbard/TNE/tne')

import consts
from evaluator_utils.convert_data2leaderboard_eval import data_conversion
from evaluator_utils.evaluate import evaluate_documents, preposition_dict


class Predictor:
    def __init__(self,
                 tokenizer,
                 gpt2: nn.Module,
                 stop_classifier: nn.Module,
                 span_predictor: nn.Module,
                 span_loss_name: str):
        self.tokenizer = tokenizer
        self.gpt2 = gpt2
        self.stop_classifier = stop_classifier
        self.span_predictor = span_predictor
        self.span_loss_name = span_loss_name

    def predict_stop(self, encoded_prompt: Dict[str, Tensor]):
        output = self.gpt2(encoded_prompt)
        stop_classifier_result = self.stop_classifier(output)
        should_stop = (stop_classifier_result.argmax() == 1).item()
        return should_stop

    def predict_span(self, encoded_prompt: Dict[str, Tensor]):
        output = self.gpt2(encoded_prompt)
        span = self.span_predictor(output)
        return span.squeeze()

    def get_closest_valid_np_id(self, span, all_spans):
        deltas = all_spans - span

        if self.span_loss_name == "L2":
            dist_2 = torch.einsum('ij,ij->i', deltas, deltas)
        else:  # self.span_loss_name == "L1"
            dist_2 = torch.sum(all_spans - span, dim=1)

        closest_valid_np_id = torch.argmin(dist_2).item()
        return closest_valid_np_id

    def predict(self, dataset: DataLoader, gold_relations_exist=True):
        gold_docs = []
        pred_docs = []

        dataset_progress_bar = tqdm(dataset)
        for document in dataset_progress_bar:
            dataset_progress_bar.set_description(f"Document {document['id']}")
            print(f"Predicting relations in document {document['id']}...")
            predicted_prepositions = torch.zeros(len(document['nps']), len(document['nps']), dtype=torch.int8)

            for noun_phrase in document['nps'].values():
                for preposition in consts.PREPOSITIONS:
                    prompt = f"{document['text']}\n\nQ: {noun_phrase['text']} {preposition}\nA: "
                    encoded_prompt = self.tokenizer(prompt, return_tensors='pt')
                    if torch.cuda.is_available():
                        encoded_prompt = encoded_prompt.to('cuda:0')

                    all_nps_spans = document['all_nps_spans'].to('cuda:0')
                    predicted_complement_ids = set()
                    should_stop = self.predict_stop(encoded_prompt)
                    i = 0
                    while i < consts.MAX_COMPLEMENTS_PER_ANCHOR_PREPOSITION and not should_stop and all_nps_spans.numel() > 0:
                        i += 1
                        next_predicted_complement_id = self.get_closest_valid_np_id(self.predict_span(encoded_prompt), all_nps_spans)
                        predicted_complement_ids.add(next_predicted_complement_id)
                        prompt += f'({all_nps_spans[next_predicted_complement_id, 0]},{all_nps_spans[next_predicted_complement_id, 1]}), '
                        all_nps_spans = torch.cat((all_nps_spans[:next_predicted_complement_id], all_nps_spans[next_predicted_complement_id+1:]))
                        encoded_prompt = self.tokenizer(prompt, return_tensors='pt')
                        if torch.cuda.is_available():
                            encoded_prompt = encoded_prompt.to('cuda:0')
                        should_stop = self.predict_stop(encoded_prompt)

                    anchor_id = int(noun_phrase['id'][2:])
                    preposition_id = preposition_dict[preposition]
                    for predicted_complement_id in predicted_complement_ids:
                        predicted_prepositions[anchor_id, predicted_complement_id] = preposition_id

            predictions = {'id': document['id'], 'predicted_prepositions': predicted_prepositions.reshape(1, -1).squeeze().tolist()}
            pred_docs.append(predictions)

            if gold_relations_exist:
                gold_relations = data_conversion(document)
                gold_docs.append(gold_relations)

        scores = None
        if gold_relations_exist:
            scores = evaluate_documents(gold_docs, pred_docs)

        return pred_docs, scores

