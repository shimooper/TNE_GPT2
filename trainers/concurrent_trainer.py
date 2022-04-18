from typing import Dict

import torch.cuda
from torch.utils.data import DataLoader
from torch import nn, Tensor, optim
from tqdm import tqdm
import json
import time

from gpt2_prompt_predictor import Predictor
from loggers.csv_logger import CSVLogger

class ConcurrentTrainer:
    def __init__(self,
                 tokenizer,
                 gpt2: nn.Module,
                 stop_classifier: nn.Module,
                 span_predictor: nn.Module,
                 stop_optimizer: optim.Optimizer,
                 span_optimizer: optim.Optimizer,
                 span_loss_name: str,
                 stop_loss: nn.Module,
                 span_loss: nn.Module,
                 span_probability: float,
                 train_span_concurrent_csv_file_path: str,
                 eval_span_concurrent_csv_file_path: str,
                 train_stop_concurrent_csv_file_path: str,
                 eval_stop_concurrent_csv_file_path: str,
                 evaluation_scores_file_path: str,
                 eval_frequency: int):

        assert (span_probability >= 0) and (span_probability <= 1)
        self.tokenizer = tokenizer
        self.gpt2 = gpt2
        self.stop_classifier = stop_classifier
        self.span_predictor = span_predictor
        self.stop_optimizer = stop_optimizer
        self.span_optimizer = span_optimizer
        self.stop_loss = stop_loss
        self.span_loss = span_loss
        self.predictor = Predictor(tokenizer, gpt2, stop_classifier, span_predictor, span_loss_name)
        self.span_probability = span_probability
        self.eval_frequency = eval_frequency
        self.train_csv_logger = CSVLogger(train_span_concurrent_csv_file_path, train_stop_concurrent_csv_file_path)
        self.eval_csv_logger = CSVLogger(eval_span_concurrent_csv_file_path, eval_stop_concurrent_csv_file_path)
        self.evaluation_scores_file_path = evaluation_scores_file_path

    def train_stop(self, encoded_prompt: Dict[str, Tensor], should_stop: int, evaluate: bool):
        output = self.gpt2(encoded_prompt)

        stop = self.stop_classifier(output)
        t = Tensor([should_stop]).long()
        if torch.cuda.is_available():
            t = t.to('cuda:0', non_blocking=True)
        stop_loss = self.stop_loss(stop, t)

        if not evaluate:
            self.gpt2.zero_grad()
            self.stop_optimizer.zero_grad()
            stop_loss.backward()
            self.stop_optimizer.step()

        return stop_loss

    def train_span(self, encoded_prompt: Dict[str, Tensor], true_spans: list, evaluate: bool):
        output = self.gpt2(encoded_prompt)
        pred_span = self.span_predictor(output)

        span_losses = [self.span_loss(pred_span[0], true_span.float()) for true_span in true_spans]
        span_loss, min_index = min((val, idx) for (idx, val) in enumerate(span_losses))
        if torch.cuda.is_available():
            span_loss.to('cuda:0', non_blocking=True)

        if not evaluate:
            self.gpt2.zero_grad()
            self.span_optimizer.zero_grad()
            span_loss.backward()
            self.span_optimizer.step()

        return span_loss, pred_span[0], min_index

    def do_epoch(self, dataset: DataLoader, epoch: int, csv_logger: CSVLogger, evaluate: bool):
        for text_id, anchor_id, prep_id, prompt, spans in dataset:
            if torch.cuda.is_available():
                spans = spans.to('cuda:0', non_blocking=True)

            if type(prompt) == tuple:
                # weird bug, sometimes prompt is returned as tuple
                assert prompt[0] and len(prompt) == 1
                prompt = prompt[0]

            num_complements = spans.shape[1]
            spans_to_check = list(spans[0])
            i = 0

            while len(spans_to_check):
                encoded_input = self.tokenizer(prompt, return_tensors='pt')
                if torch.cuda.is_available():
                    encoded_input = encoded_input.to('cuda:0')

                stop_loss = self.train_stop(encoded_input, 0, evaluate)

                csv_logger.log_stop(epoch, text_id[0], anchor_id[0], prep_id[0],
                                               num_complements, i, stop_loss.item())

                if torch.rand(1) < self.span_probability:
                    span_loss, pred_span, span_checked_index = self.train_span(encoded_input,
                                                                               spans_to_check, evaluate)
                    span_checked = spans_to_check[span_checked_index]

                    span_start, span_end = span_checked[0].item(), span_checked[1].item()
                    pred_start, pred_end = pred_span[0].item(), pred_span[1].item()
                    csv_logger.log_span(epoch, text_id[0], anchor_id[0], prep_id[0],
                                                   span_start, span_end,
                                                   pred_start, pred_end,
                                                   span_loss.item())
                    del spans_to_check[span_checked_index]

                else:
                    span_checked = spans_to_check.pop(0)

                prompt += f' ({span_checked[0].item()},{span_checked[1].item()})'
                i += 1

            encoded_input = self.tokenizer(prompt, return_tensors='pt')
            if torch.cuda.is_available():
                encoded_input = encoded_input.to('cuda:0')
            stop_loss = self.train_stop(encoded_input, 1, evaluate)
            csv_logger.log_stop(epoch, text_id[0], anchor_id[0], prep_id[0],
                                           num_complements, num_complements, stop_loss.item())

    def train(self, train_dataset: DataLoader, eval_ds_for_loss: DataLoader, eval_ds_for_metrics: DataLoader, epochs: int):
        epochs_progress_bar = tqdm(range(epochs))
        evaluation_labeled_f1_per_epoch = []
        evaluation_unlabeled_f1_per_epoch = []
        for epoch in epochs_progress_bar:
            self.span_predictor.train()
            self.stop_classifier.train()
            self.gpt2.train()
            epochs_progress_bar.set_description("Epoch %d" % (epoch+1))

            print(f"Concurrent_trainer - training on train dataset... epoch {epoch+1}")
            start = time.time()
            self.do_epoch(train_dataset, epoch, self.train_csv_logger, False)
            end = time.time()
            print(f"Concurrent_trainer - training epoch {epoch+1} took {end-start} seconds")

            with torch.no_grad():
                self.span_predictor.eval()
                self.stop_classifier.eval()
                self.gpt2.eval()

                print(f"Concurrent_trainer - evaluating loss on dev dataset... after epoch {epoch+1}")
                start = time.time()
                self.do_epoch(eval_ds_for_loss, epoch, self.eval_csv_logger, True)
                end = time.time()
                print(f"Concurrent_trainer - evaluation loss after epoch {epoch + 1} took {end - start} seconds")

                if ((epoch + 1) % self.eval_frequency) == 0:
                    print(f"Concurrent_trainer - evaluating metrics on dev dataset... after epoch {epoch+1}")
                    start = time.time()
                    _, scores = self.predictor.predict(eval_ds_for_metrics)
                    end = time.time()
                    print(f"Concurrent_trainer - evaluation metrics on dev dataset after epoch {epoch + 1} took {end - start} seconds")

                    print(f"Concurrent_trainer - scores on dev dataset after epoch {epoch + 1}: {scores}")
                    with open(self.evaluation_scores_file_path, "a") as f:
                        f.write(f"After concurrent training epoch {epoch+1}: {json.dumps(scores)}\n")
                    evaluation_labeled_f1_per_epoch.append(scores['labeled_f1'])
                    evaluation_unlabeled_f1_per_epoch.append(scores['unlabeled_f1'])

        if epochs > 0 and evaluation_labeled_f1_per_epoch:
            best_epoch_for_labeled_f1 = evaluation_labeled_f1_per_epoch.index(max(evaluation_labeled_f1_per_epoch)) + 1
            best_epoch_for_unlabeled_f1 = evaluation_unlabeled_f1_per_epoch.index(max(evaluation_unlabeled_f1_per_epoch)) + 1
            with open(self.evaluation_scores_file_path, "a") as f:
                f.write(f"Best epoch for labeled f1 after concurrent training: {best_epoch_for_labeled_f1}\n")
                f.write(f"Best epoch for unlabeled f1 after concurrent training: {best_epoch_for_unlabeled_f1}\n")
