import os
import pandas as pd
import consts
import pickle
import numpy as np
import argparse

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr, pearsonr

from datasets.next_span_prediction_ds import NextSpanPredictionDataset

FONT_SIZE = 24

def get_num_epochs(df):
    num_epochs = max(df.epoch_id)+1
    return num_epochs

def get_epoch_filter(df, num_epochs):
    epoch_filter = df.epoch_id == num_epochs-1
    return epoch_filter

def get_losses_by_epoch(df, num_epochs, output_dir, subdir_name):
    ###### median_losses_by_epoch.html
    losses_by_epoch = [df.loss[df.epoch_id == i] for i in range(num_epochs)]
    median_loss_by_epoch = [(i, losses.median()) for i, losses in enumerate(losses_by_epoch)]
    x, y = list(zip(*median_loss_by_epoch))
    df = pd.DataFrame({"Epoch": x, "Median Loss": y})
    k = list(df.keys())
    fig = px.scatter(df, x=k[0], y=k[1])
    fig.update_layout(font=dict(size=FONT_SIZE))
    fig.write_html(os.path.join(output_dir, subdir_name, "median_losses_by_epoch.html"))

    ###### avg_losses_by_epoch.html
    avg_loss_by_epoch = [(i, losses.mean()) for i, losses in enumerate(losses_by_epoch)]
    x, y = list(zip(*avg_loss_by_epoch))
    df = pd.DataFrame({"Epoch": x, "Average Loss": y})
    k = list(df.keys())
    fig = px.scatter(df, x=k[0], y=k[1])
    fig.update_layout(font=dict(size=FONT_SIZE))
    fig.write_html(os.path.join(output_dir, subdir_name, "avg_losses_by_epoch.html"))

def get_prob_stop_by_curr_num_complements(df, epoch_filter, output_dir, subdir_name):
    df['should_continue'] = df['curr_num_complements'] < df['num_complements']
    df['prob'] = np.abs(df['should_continue'] - np.exp(-df['loss']))

    df2 = df[epoch_filter].groupby(['prep_id', "curr_num_complements"]).mean('prob').reset_index()

    fig = px.line(df2, x='curr_num_complements', y='prob', color='prep_id')
    fig.update_layout(font=dict(size=FONT_SIZE))
    fig.write_html(os.path.join(output_dir, subdir_name, "avg_prob_stop_by_curr_num_complements.html"))

    df2 = df[epoch_filter].groupby(['prep_id', "curr_num_complements"]).median('prob').reset_index()
    fig = px.line(df2, x='curr_num_complements', y='prob', color='prep_id')
    fig.update_layout(font=dict(size=FONT_SIZE))
    fig.write_html(os.path.join(output_dir, subdir_name, "median_prob_stop_by_curr_num_complements.html"))

def get_losses_by_prep_id(df, epoch_filter, preps_dict, output_dir, is_span):
    prep_ids = set(df.prep_id[epoch_filter])
    if is_span:
        subdir_name = "span"
        losses_by_prep_id = [df.loss[epoch_filter][df.prep_id == prep_id][df.span_start != -1.0] for prep_id in prep_ids]
    else:
        subdir_name = "stop"
        losses_by_prep_id = [df.loss[epoch_filter][df.prep_id == prep_id] for prep_id in prep_ids]
    
    avg_losses_by_prep = [(prep_id, losses.mean()) if len(losses) else None for prep_id, losses in zip(prep_ids, losses_by_prep_id)]
    avg_losses_by_prep = [x for x in avg_losses_by_prep if x]
    avg_losses_by_prep.sort(key=lambda x:x[1], reverse=True)

    median_losses_by_prep = [(prep_id, losses.median()) if len(losses) else None for prep_id, losses in zip(prep_ids, losses_by_prep_id)]
    median_losses_by_prep = [x for x in median_losses_by_prep if x]
    median_losses_by_prep.sort(key=lambda x:x[1], reverse=True)

    ###### avg_losses_by_prep.html
    x, y = list(zip(*avg_losses_by_prep))
    preps = [preps_dict[y] for y in x]
    preps /= sum(preps)

    df = pd.DataFrame({"True preposition": x, "Average loss according to last epoch": y, "count": preps})
    k = list(df.keys())

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(name='Average loss according to last epoch', x=x, y=y, offsetgroup=1))
    fig.add_trace(go.Bar(name='Probability of prepositions in training data', x=x, y=preps, offsetgroup=2), secondary_y=True)
    fig.update_xaxes(title_text='Preposition')
    fig.update_yaxes(title_text='Average loss', secondary_y=False)
    fig.update_yaxes(title_text='Preposition probability', secondary_y=True)#range=[0,1]
    fig.update_layout(barmode='group')
    fig.update_layout(font=dict(size=FONT_SIZE))
    fig.write_html(os.path.join(output_dir, subdir_name, "avg_losses_by_prep.html"))

    ###### median_losses_by_prep.html
    x, y = list(zip(*median_losses_by_prep))

    preps = [preps_dict[z] for z in x]
    preps /= sum(preps)

    df = pd.DataFrame({"True preposition": x, "Median loss according to last epoch": y, "count": preps})
    k = list(df.keys())

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(name='Median loss according to last epoch', x=x, y=y, offsetgroup=1))
    fig.add_trace(go.Bar(name='Probability of prepositions in training data', x=x, y=preps, offsetgroup=2), secondary_y=True)
    fig.update_xaxes(title_text='Preposition')
    fig.update_yaxes(title_text='Median loss', secondary_y=False)
    fig.update_yaxes(title_text='Preposition probability', secondary_y=True)#range=[0,1]
    fig.update_layout(barmode='group')
    fig.update_layout(font=dict(size=FONT_SIZE))
    fig.write_html(os.path.join(output_dir, subdir_name, "median_losses_by_prep.html"))
    pd.DataFrame({"y": y, "preps": preps}).to_csv("y_preps.csv")
    print(y)
    print(preps)
    print("spearmanr:", spearmanr(y, preps))
    print("pearsonr:", pearsonr(y, preps))

def get_central_start_index(ds):
    return ds.nps['first_token'].mean(), ds.nps['first_token'].median()

def get_losses_by_span_start(ds, df, epoch_filter, output_dir, subdir_name):
    span_starts = set(df.span_start[epoch_filter])
    losses_by_span_start = [df.loss[df.span_start == span_start][epoch_filter] for span_start in span_starts]

    avg_losses_by_span_start = [(x, y.mean()) for x, y in zip(span_starts, losses_by_span_start)]
    avg_losses_by_span_start.sort(key=lambda x:x[1], reverse=True)

    median_losses_by_span_start = [(x, y.median()) for x, y in zip(span_starts, losses_by_span_start)]
    median_losses_by_span_start.sort(key=lambda x:x[1], reverse=True)

    avg, median = get_central_start_index(ds)

    ###### avg_losses_by_span_start.html
    x, y = list(zip(*avg_losses_by_span_start))
    df = pd.DataFrame({"True NP span start index": x, "Average loss according to last epoch": y})
    k = list(df.keys())
    fig = px.scatter(df, x=k[0], y=k[1])
    fig.add_vline(x=median, line_width=3, line_dash="dash", line_color="green", annotation_text="median start index", annotation_position="top left")
    fig.update_layout(font=dict(size=FONT_SIZE))
    fig.write_html(os.path.join(output_dir, subdir_name, "avg_losses_by_span_start.html"))

    ###### median_losses_by_span_start.html
    x, y = list(zip(*median_losses_by_span_start))
    df = pd.DataFrame({"True NP span start index": x, "Median loss according to last epoch": y})
    k = list(df.keys())
    fig = px.scatter(df, x=k[0], y=k[1])
    fig.add_vline(x=median, line_width=3, line_dash="dash", line_color="green", annotation_text="median start index", annotation_position="top left")
    fig.update_layout(font=dict(size=FONT_SIZE))
    fig.write_html(os.path.join(output_dir, subdir_name, "median_losses_by_span_start.html"))

def get_prep_histograms(ds, output_dir):
    ###### prep_histograms.html
    merged = pd.merge(ds.relations, ds.nps, how='left', left_on=['complement', 'id'], right_on=['np_id', 'id'])
    merged['len'] = merged['last_token'] - merged['first_token'] + 1
    fig = px.histogram(merged, x="first_token", color="preposition", histnorm='probability')
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.5)
    fig.update_layout(font=dict(size=FONT_SIZE))
    fig.write_html(os.path.join(output_dir, "complement_first_token_histograms_per_prep.html"))

    fig = px.histogram(merged, x="last_token", color="preposition", histnorm='probability')
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.5)
    fig.update_layout(font=dict(size=FONT_SIZE))
    fig.write_html(os.path.join(output_dir, "complement_last_token_histograms_per_prep.html"))

    fig = px.histogram(merged, x="len", color="preposition", histnorm='probability')
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.5)
    fig.update_layout(font=dict(size=FONT_SIZE))
    fig.write_html(os.path.join(output_dir, "complement_len_histograms_per_prep.html"))

def main():
    PREPS_PKL = "preps.pkl"

    ds_path = "../../data/train.jsonl"
    ds = NextSpanPredictionDataset(ds_path, consts.PREPOSITIONS)

    # Generate prepositions dict
    if not os.path.isfile(PREPS_PKL):
        preps_dict = {p : ds.relations['preposition'][ds.relations['preposition'] == p].count() for p in consts.PREPOSITIONS}
        with open(PREPS_PKL, "wb") as f:
            pickle.dump(preps, f)
    else:
        print("Loading preps.pkl")
        with open(PREPS_PKL, "rb") as f:
            preps_dict = pickle.load(f)

    parser = argparse.ArgumentParser(description='Create error analysis graphs.')
    parser.add_argument('--span_csv_path', type=str, default=None,
                        help='Span CSV file path')
    parser.add_argument('--stop_csv_path', type=str, default=None,
                        help='Stop CSV file path')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for graph html files', required=True)
    parser.add_argument('--num_epochs', type=int, default=-1,
                        help='Number of epochs to consider when creating the graphs (defaults to all epochs)')

    args = parser.parse_args()

    span_csv_path = args.span_csv_path
    stop_csv_path = args.stop_csv_path
    output_dir = args.output_dir
    num_epochs_param = args.num_epochs

    # span
    if span_csv_path:
        os.makedirs(os.path.join(output_dir, "span"), exist_ok=True)
        df = pd.read_csv(span_csv_path)
        if num_epochs_param != -1:
            num_epochs = num_epochs_param
        else:
            num_epochs = get_num_epochs(df)
        get_losses_by_epoch(df, num_epochs, output_dir, "span")
        epoch_filter = get_epoch_filter(df, num_epochs)
        get_losses_by_prep_id(df, epoch_filter, preps_dict, output_dir, True)
        get_losses_by_span_start(ds, df, epoch_filter, output_dir, "span")

    # stop
    if stop_csv_path:
        os.makedirs(os.path.join(output_dir, "stop"), exist_ok=True)
        df = pd.read_csv(stop_csv_path)
        if num_epochs_param != -1:
            num_epochs = num_epochs_param
        else:
            num_epochs = get_num_epochs(df)
        get_losses_by_epoch(df, num_epochs, output_dir, "stop")
        epoch_filter = get_epoch_filter(df, num_epochs)
        get_losses_by_prep_id(df, epoch_filter, preps_dict, output_dir, False)
        get_prob_stop_by_curr_num_complements(df, epoch_filter, output_dir, "stop")

    get_prep_histograms(ds, output_dir)

if __name__ == "__main__":
    main()
