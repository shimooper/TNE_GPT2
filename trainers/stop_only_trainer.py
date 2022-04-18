from typing import Dict

import torch.cuda
from torch.utils.data import DataLoader
from torch import nn, Tensor, optim
from tqdm import tqdm
import torch
import time
from .util import choose_span_sample
from gpt2_prompt_predictor import Predictor
from loggers.csv_logger import CSVLogger


class StopOnlyTrainer:
    def __init__(self,
                 tokenizer,
                 gpt2: nn.Module,
                 stop_predictor: nn.Module,
                 stop_optimizer: optim.Optimizer,
                 stop_loss: nn.Module,
                 train_stop_only_csv_file_path: str,
                 eval_stop_only_csv_file_path: str):
        self.tokenizer = tokenizer
        self.gpt2 = gpt2
        self.stop_classifier = stop_predictor
        self.stop_optimizer = stop_optimizer
        self.stop_loss = stop_loss
        self.train_csv_logger = CSVLogger(None, train_stop_only_csv_file_path)
        self.eval_csv_logger = CSVLogger(None, eval_stop_only_csv_file_path)

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

    def do_epoch(self, dataset: DataLoader, epoch: int, csv_logger: CSVLogger, evaluate: bool):
        for text_id, anchor_id, prep_id, prompt, spans in dataset:
            chosen = choose_span_sample(spans.shape[1])
            if torch.cuda.is_available():
                spans = spans.to('cuda:0', non_blocking=True)

            if type(prompt) == tuple:
                # weird bug, sometimes prompt is returned as tuple
                assert prompt[0] and len(prompt) == 1
                prompt = prompt[0]

            num_complements = spans.shape[1]
            for i in range(spans.shape[1]):
                if chosen == i:
                    encoded_input = self.tokenizer(prompt, return_tensors='pt')
                    if torch.cuda.is_available():
                        encoded_input = encoded_input.to('cuda:0')

                    stop_loss = self.train_stop(encoded_input, 0, evaluate)
                    csv_logger.log_stop(epoch, text_id[0], anchor_id[0], prep_id[0],
                                                   num_complements, i, stop_loss.item())

                prompt += f' ({spans[0][i][0].item()},{spans[0][i][1].item()})'

            encoded_input = self.tokenizer(prompt, return_tensors='pt')

            if torch.cuda.is_available():
                encoded_input = encoded_input.to('cuda:0')

            stop_loss = self.train_stop(encoded_input, 1, evaluate)
            csv_logger.log_stop(epoch, text_id[0], anchor_id[0], prep_id[0],
                                           num_complements, num_complements, stop_loss.item())

    def train(self, train_dataset: DataLoader, eval_ds_for_loss: DataLoader, epochs: int):
        epochs_progress_bar = tqdm(range(epochs))
        for epoch in epochs_progress_bar:
            self.stop_classifier.train()
            self.gpt2.train()
            epochs_progress_bar.set_description("Epoch %d" % (epoch+1))

            print(f"StopOnly_trainer - training on train dataset... epoch {epoch+1}")
            start = time.time()
            self.do_epoch(train_dataset, epoch, self.train_csv_logger, False)
            end = time.time()
            print(f"StopOnly_trainer - training epoch {epoch+1} took {end-start} seconds")

            with torch.no_grad():
                self.stop_classifier.eval()
                self.gpt2.eval()

                print(f"StopOnly_trainer - evaluating loss on dev dataset... after epoch {epoch + 1}")
                start = time.time()
                self.do_epoch(eval_ds_for_loss, epoch, self.eval_csv_logger, True)
                end = time.time()
                print(f"StopOnly_trainer - evaluation loss after epoch {epoch + 1} took {end - start} seconds")


