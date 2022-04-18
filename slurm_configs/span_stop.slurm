#! /bin/sh
#SBATCH --job-name=span_stop
#SBATCH --output=slurm_configs/slurm_out/span_stop/span_stop.out
#SBATCH --error=slurm_configs/slurm_out/span_stop/span_stop.err
#SBATCH --partition=studentbatch
#SBATCH --time=720
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python /home/joberant/NLP_2122/idangrosbard/TNE/tne/custom/train_gpt_prompt.py /home/joberant/NLP_2122/idangrosbard/TNE/data/train.jsonl /home/joberant/NLP_2122/idangrosbard/TNE/data/dev.jsonl /home/joberant/NLP_2122/idangrosbard/TNE/tne/custom/weights/span_stop --span_loss L2 --span_epochs 40 --stop_epochs 0 --concurrent_epochs 0 --span_prob 0.8 --stop_as_negative_span
