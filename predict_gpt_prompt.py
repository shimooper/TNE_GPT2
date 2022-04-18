import torch
import os.path
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from torch import nn, Tensor, optim
from datasets import PredictionDataset
from argparse import ArgumentParser, Namespace
from models import gpt2_prompt_models
from gpt2_prompt_predictor import Predictor
from models.gpt2_prompt_models.prompt_encoding_reduction import LastTokenReduction
import consts
import json


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--ds_test_path', type=str, help='Path to the test dataset file')
    parser.add_argument('--model_weights_dir', type=str, help='Path of the saved model')
    parser.add_argument('--predictions_output_dir', type=str, help='Path to save the predictions')
    parser.add_argument('--gold_relations_exist', type=bool, default=False, help='Whether gold relations exist in ds_test_path')

    parser.add_argument('--stop_layers', type=int, default=3, help='num of hidden layers for the stop predictor')
    parser.add_argument('--span_layers', type=int, default=3, help='num of hidden layers for the span predictor')

    parser.add_argument('--mlp_hidden_dim', type=int, default=512, help='dim of hidden layers in MLPs')

    parser.add_argument('--gpt_pretraining', type=str, default='gpt2', help='Type of pretraining for gpt-2')
    parser.add_argument('--prompt_reduction', type=str, default='last_token', help='How to reduce the encoding of the entire prompt to a single vector')

    parser.add_argument('--span_loss', type=str, default='L2')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model_paths = {"gpt2": os.path.join(args.model_weights_dir, 'gpt2.pth'),
                   "stop": os.path.join(args.model_weights_dir, 'stop.pth'),
                   "span": os.path.join(args.model_weights_dir, 'span.pth')}

    print("Initializing data loader...")
    test_ds = DataLoader(PredictionDataset(args.ds_test_path, False), num_workers=consts.NUM_WORKERS, batch_size=None)
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_pretraining)

    if args.prompt_reduction == 'last_token':
        reduction = LastTokenReduction()

    prompt_encoder = gpt2_prompt_models.PromptEncoder(reduction, args.gpt_pretraining)
    if os.path.isfile(model_paths["gpt2"]):
        print("Found gpt2 model file. Loading it.")
        prompt_encoder.load_state_dict(torch.load(model_paths["gpt2"]))

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

    span_predictor = gpt2_prompt_models.SpanPredictor(
        consts.GPT2_HIDDEN,
        args.mlp_hidden_dim,
        args.span_layers)
    if os.path.isfile(model_paths["span"]):
        print("Found span model file. Loading it.")
        span_predictor.load_state_dict(torch.load(model_paths["span"]))

    if torch.cuda.is_available():
        span_predictor.cuda()

    print("Initializing predictor...")
    predictor = Predictor(
        tokenizer, prompt_encoder,
        stop_classifier, span_predictor, args.span_loss)

    print("Predicting...")
    with torch.no_grad():
        predictions, scores = predictor.predict(test_ds, args.gold_relations_exist)

    predictions_output_file = os.path.join(args.predictions_output_dir, "predictions.txt")
    with open(predictions_output_file, "w") as f:
        f.write(json.dumps(predictions))
    if args.gold_relations_exist:
        predictions_score_file = os.path.join(args.predictions_output_dir, "predictions_scores.txt")
        with open(predictions_score_file, "w") as f:
            f.write(scores)
    print("Done predicting!")
