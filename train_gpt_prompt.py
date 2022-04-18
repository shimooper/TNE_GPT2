import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from torch import nn, Tensor, optim
from datasets import NextSpanPredictionDataset, PredictionDataset
from argparse import ArgumentParser, Namespace
from models import gpt2_prompt_models
from models.gpt2_prompt_models.prompt_encoding_reduction import LastTokenReduction
import consts
import trainers

import sys
sys.path.append('/home/joberant/NLP_2122/idangrosbard/TNE')
sys.path.append('/home/joberant/NLP_2122/idangrosbard/TNE/tne')

model_paths = {}

def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('ds_train_path', type=str, help='Path to the training dataset file')
    parser.add_argument('ds_eval_path', type=str, help='Path to the evaluation dataset file')
    parser.add_argument('model_weights_output_dir', type=str, help='Path where to save the model')

    parser.add_argument('--stop_layers', type=int, default=3, help='num of hidden layers for the stop predictor')
    parser.add_argument('--span_layers', type=int, default=3, help='num of hidden layers for the span predictor')

    parser.add_argument('--mlp_hidden_dim', type=int, default=512, help='dim of hidden layers in MLPs')

    parser.add_argument('--gpt_pretraining', type=str, default='gpt2', help='Type of pretraining for gpt-2')
    parser.add_argument('--prompt_reduction', type=str, default='last_token', help='How to reduce the encoding of the entire prompt to a single vector')

    parser.add_argument('--span_loss', type=str, default='L2')
    parser.add_argument('--span_epochs', type=int, default=4, help='num epochs to train span predictor')
    parser.add_argument('--stop_epochs', type=int, default=4, help='num epochs to train stop predictor')
    parser.add_argument('--concurrent_epochs', type=int, default=4, help='num epochs to train both models concurrently')
    parser.add_argument('--span_prob', type=float, default=0.5, help='probability of training the span predictor')
    parser.add_argument('--stop_as_negative_span', action='store_true', help='probability of training the span predictor')

    parser.add_argument('--eval_frequency', type=int, default=10,
                        help='frequency of running the evaluation code (in epochs)')
    parser.add_argument('--only_train_random_weights', action='store_true',
                        help='If to train entire GPT2 or just the random initialized weights')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model_paths["gpt2"] = os.path.join(args.model_weights_output_dir, 'gpt2.pth')
    model_paths["stop"] = os.path.join(args.model_weights_output_dir, 'stop.pth')
    model_paths["span"] = os.path.join(args.model_weights_output_dir, 'span.pth')
    evaluation_scores_path = os.path.join(args.model_weights_output_dir, 'evaluation_scores.txt')
    Path(args.model_weights_output_dir).mkdir(parents=True, exist_ok=True)

    print("Initializing data loader...")
    train_ds = DataLoader(NextSpanPredictionDataset(args.ds_train_path, consts.PREPOSITIONS),
                          num_workers=consts.NUM_WORKERS, batch_size=consts.BATCH_SIZE, shuffle=True)
    eval_ds_for_loss = DataLoader(NextSpanPredictionDataset(args.ds_eval_path, consts.PREPOSITIONS),
                          num_workers=consts.NUM_WORKERS, batch_size=consts.BATCH_SIZE, shuffle=True)
    eval_ds_for_metrics = DataLoader(PredictionDataset(args.ds_eval_path, True), num_workers=consts.NUM_WORKERS, batch_size=None)
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_pretraining)

    if args.prompt_reduction == 'last_token':
        reduction = LastTokenReduction()

    prompt_encoder = gpt2_prompt_models.PromptEncoder(reduction, args.gpt_pretraining)
    if os.path.isfile(model_paths["gpt2"]):
        print("Found gpt2 model file. Loading it.")
        prompt_encoder.load_state_dict(torch.load(model_paths["gpt2"]))
    if args.only_train_random_weights:
        trainers.train_only_necassary_gpt2_weights(prompt_encoder)

    for m in prompt_encoder.modules():
        m.requires_grad_(True)

    if torch.cuda.is_available():
        prompt_encoder.cuda()

    stop_classifier = gpt2_prompt_models.StopPredictor(
        consts.GPT2_HIDDEN,
        args.mlp_hidden_dim,
        args.stop_layers)
    if os.path.isfile(model_paths["stop"]):
        print("Found stop model file. Loading it.")
        stop_classifier.load_state_dict(torch.load(model_paths["stop"]))

    if torch.cuda.is_available():
        stop_classifier.cuda()

    stop_optimizer = optim.Adam([p for p in stop_classifier.parameters()] + [p for p in prompt_encoder.parameters()], lr=1e-5)

    stop_loss = nn.CrossEntropyLoss()

    span_predictor = gpt2_prompt_models.SpanPredictor(
        consts.GPT2_HIDDEN,
        args.mlp_hidden_dim,
        args.span_layers)
    if os.path.isfile(model_paths["span"]):
        print("Found span model file. Loading it.")
        span_predictor.load_state_dict(torch.load(model_paths["span"]))

    if torch.cuda.is_available():
        span_predictor.cuda()

    span_optimizer = optim.Adam([p for p in span_predictor.parameters()] + [p for p in prompt_encoder.parameters()], lr=1e-5)

    if args.span_loss == 'L2':
        span_loss = nn.MSELoss()
    elif args.span_loss == 'L1':
        span_loss = nn.L1Loss()

    print("Initializing trainer...")
    train_span_only_csv_file_path = os.path.join(args.model_weights_output_dir, "train_span_only.csv")
    eval_span_only_csv_file_path = os.path.join(args.model_weights_output_dir, "eval_span_only.csv")

    span_trainer = trainers.SpanOnlyTrainer(
        tokenizer, prompt_encoder, span_predictor,
        span_optimizer, args.span_loss, span_loss, args.stop_as_negative_span,
        train_span_only_csv_file_path, eval_span_only_csv_file_path, evaluation_scores_path, args.eval_frequency)

    train_stop_only_csv_file_path = os.path.join(args.model_weights_output_dir, "train_stop_only.csv")
    eval_stop_only_csv_file_path = os.path.join(args.model_weights_output_dir, "eval_stop_only.csv")

    stop_trainer = trainers.StopOnlyTrainer(
        tokenizer, prompt_encoder,
        stop_classifier, stop_optimizer, stop_loss,
        train_stop_only_csv_file_path, eval_stop_only_csv_file_path)

    train_span_concurrent_csv_file_path = os.path.join(args.model_weights_output_dir, "train_span_concurrent.csv")
    eval_span_concurrent_csv_file_path = os.path.join(args.model_weights_output_dir, "eval_span_concurrent.csv")
    train_stop_concurrent_csv_file_path = os.path.join(args.model_weights_output_dir, "train_stop_concurrent.csv")
    eval_stop_concurrent_csv_file_path = os.path.join(args.model_weights_output_dir, "eval_stop_concurrent.csv")

    concurrent_trainer = trainers.ConcurrentTrainer(
        tokenizer, prompt_encoder,
        stop_classifier, span_predictor,
        stop_optimizer, span_optimizer,
        args.span_loss, stop_loss, span_loss, args.span_prob,
        train_span_concurrent_csv_file_path, eval_span_concurrent_csv_file_path,
        train_stop_concurrent_csv_file_path, eval_stop_concurrent_csv_file_path,
        evaluation_scores_path, args.eval_frequency)

    print("Training...")
    span_trainer.train(train_ds, eval_ds_for_loss, eval_ds_for_metrics, args.span_epochs)

    if not args.stop_as_negative_span:
        stop_trainer.train(train_ds, eval_ds_for_loss, args.stop_epochs)
    
    concurrent_trainer.train(train_ds, eval_ds_for_loss, eval_ds_for_metrics, args.concurrent_epochs)
    print("Done training!")

    torch.save(concurrent_trainer.gpt2.state_dict(), model_paths["gpt2"])
    if not args.stop_as_negative_span:
        torch.save(concurrent_trainer.stop_classifier.state_dict(), model_paths["stop"])
    torch.save(concurrent_trainer.span_predictor.state_dict(), model_paths["span"])
