from typing import Dict

import torch.cuda
from torch.utils.data import DataLoader
from torch import nn, Tensor, optim
from tqdm import tqdm
import json
import torch
import time
from models.gpt2_prompt_models import SpanStopPredictor
from gpt2_prompt_predictor import Predictor
from loggers.csv_logger import CSVLogger


class SpanOnlyTrainer:
    def __init__(self,
                 tokenizer,
                 gpt2: nn.Module,
                 span_predictor: nn.Module,
                 span_optimizer: optim.Optimizer,
                 span_loss_name: str,
                 span_loss: nn.Module,
                 stop_as_negative_span: bool,
                 train_span_only_csv_file_path: str,
                 eval_span_only_csv_file_path: str,
                 evaluation_scores_file_path: str,
                 eval_frequency: int):
        self.tokenizer = tokenizer
        self.gpt2 = gpt2
        self.span_predictor = span_predictor
        self.span_optimizer = span_optimizer
        self.span_loss = span_loss
        self.predictor = Predictor(tokenizer, gpt2, SpanStopPredictor(self.span_predictor), span_predictor, span_loss_name)
        self.stop_as_negative_span = stop_as_negative_span
        self.train_csv_logger = CSVLogger(train_span_only_csv_file_path, None)
        self.eval_csv_logger = CSVLogger(eval_span_only_csv_file_path, None)
        self.evaluation_scores_file_path = evaluation_scores_file_path
        self.eval_frequency = eval_frequency

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

        """
        self.gpt2.zero_grad()
        self.span_optimizer.zero_grad()
        a1 = [x.clone() for x in self.gpt2.parameters()] + [x.clone() for x in self.span_predictor.parameters()]
        span_loss.backward()
        self.span_optimizer.step()
        b1 = [x.clone() for x in self.gpt2.parameters()] + [x.clone() for x in self.span_predictor.parameters()]
        for a, b in zip(a1, b1):
            x = torch.equal(a,b)
            y = torch.equal(a.data,b.data)
            if not x:
                print(x)
            if not y:
                print(y)
        """

        return span_loss, pred_span[0], min_index

    def do_epoch(self, dataset: DataLoader, epoch: int, csv_logger: CSVLogger, evaluate: bool):
        for text_id, anchor_id, prep_id, prompt, spans in dataset:
            if torch.cuda.is_available():
                spans = spans.to('cuda:0', non_blocking=True)

            if type(prompt) == tuple:
                # weird bug, sometimes prompt is returned as tuple
                assert prompt[0] and len(prompt) == 1
                prompt = prompt[0]

            spans_to_check = list(spans[0])  # Or 1?
            while len(spans_to_check):
                encoded_input = self.tokenizer(prompt, return_tensors='pt')
                if torch.cuda.is_available():
                    encoded_input = encoded_input.to('cuda:0')

                span_loss, pred_span, span_checked_index = self.train_span(encoded_input, spans_to_check, evaluate)
                span_checked = spans_to_check[span_checked_index]

                span_start, span_end = span_checked[0].item(), span_checked[1].item()
                pred_start, pred_end = pred_span[0].item(), pred_span[1].item()
                csv_logger.log_span(epoch, text_id[0], anchor_id[0], prep_id[0],
                                               span_start, span_end,
                                               pred_start, pred_end,
                                               span_loss.item())
                del spans_to_check[span_checked_index]

                prompt += f' ({span_checked[0].item()},{span_checked[1].item()})'

            if self.stop_as_negative_span:
                encoded_input = self.tokenizer(prompt, return_tensors='pt')
                end_span = -torch.ones(2)

                if torch.cuda.is_available():
                    encoded_input = encoded_input.to('cuda:0')
                    end_span = end_span.to('cuda:0')
                end_span.unsqueeze(0).repeat(spans.shape[0], 1)

                span_loss, pred_span, span_checked_index = self.train_span(encoded_input, [end_span], evaluate)
                span_checked = spans_to_check[span_checked_index]

                span_start, span_end = span_checked[0].item(), span_checked[1].item()
                pred_start, pred_end = pred_span[0].item(), pred_span[1].item()
                csv_logger.log_span(epoch, text_id[0], anchor_id[0], prep_id[0],
                                               span_start, span_end,
                                               pred_start, pred_end,
                                               span_loss.item())

    def train(self, train_dataset: DataLoader, eval_ds_for_loss: DataLoader, eval_ds_for_metrics: DataLoader, epochs: int):
        epochs_progress_bar = tqdm(range(epochs))
        evaluation_labeled_f1_per_epoch = []
        evaluation_unlabeled_f1_per_epoch = []
        for epoch in epochs_progress_bar:
            self.span_predictor.train()
            self.gpt2.train()
            epochs_progress_bar.set_description("Epoch %d" % (epoch+1))

            print(f"SpanOnly_trainer - training on train dataset... epoch {epoch+1}")
            start = time.time()
            self.do_epoch(train_dataset, epoch, self.train_csv_logger, False)
            end = time.time()
            print(f"SpanOnly_trainer - training epoch {epoch+1} took {end-start} seconds")

            with torch.no_grad():
                self.span_predictor.eval()
                self.gpt2.eval()

                print(f"SpanOnly_trainer - evaluating loss on dev dataset... after epoch {epoch + 1}")
                start = time.time()
                self.do_epoch(eval_ds_for_loss, epoch, self.eval_csv_logger, True)
                end = time.time()
                print(f"SpanOnly_trainer - evaluation loss after epoch {epoch + 1} took {end - start} seconds")

                if self.stop_as_negative_span and ((epoch + 1) % self.eval_frequency == 0):
                    print(f"SpanOnly_trainer (with stop_as_negative_span) - evaluating metrics on dev dataset... after epoch {epoch+1}")
                    start = time.time()
                    _, scores = self.predictor.predict(eval_ds_for_metrics)
                    end = time.time()
                    print(f"SpanOnly_trainer (with stop_as_negative_span) - evaluation metrics on dev dataset after epoch {epoch + 1} took {end - start} seconds")

                    print(f"SpanOnly_trainer (with stop_as_negative_span) - scores on dev dataset after epoch {epoch + 1}: {scores}")
                    with open(self.evaluation_scores_file_path, "a") as f:
                        f.write(f"After span-only training epoch {epoch+1}: {json.dumps(scores)}\n")
                    evaluation_labeled_f1_per_epoch.append(scores['labeled_f1'])
                    evaluation_unlabeled_f1_per_epoch.append(scores['unlabeled_f1'])
        
        if epochs > 0 and self.stop_as_negative_span and evaluation_labeled_f1_per_epoch:
            best_epoch_for_labeled_f1 = evaluation_labeled_f1_per_epoch.index(max(evaluation_labeled_f1_per_epoch)) + 1
            best_epoch_for_unlabeled_f1 = evaluation_unlabeled_f1_per_epoch.index(max(evaluation_unlabeled_f1_per_epoch)) + 1
            with open(self.evaluation_scores_file_path, "a") as f:
                f.write(f"Best epoch for labeled f1 after span-only training: {best_epoch_for_labeled_f1}\n")
                f.write(f"Best epoch for unlabeled f1 after span-only training: {best_epoch_for_unlabeled_f1}\n")
