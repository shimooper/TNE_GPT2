#! /bin/sh
#SBATCH --job-name=L1
#SBATCH --output=slurm_configs/slurm_out/L1/L1.out
#SBATCH --error=slurm_configs/slurm_out/L1/L1.err
#SBATCH --partition=studentbatch
#SBATCH --time=720
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python /home/joberant/NLP_2122/idangrosbard/TNE/tne/custom/train_gpt_prompt.py /home/joberant/NLP_2122/idangrosbard/TNE/data/train.jsonl /home/joberant/NLP_2122/idangrosbard/TNE/data/dev.jsonl /home/joberant/NLP_2122/idangrosbard/TNE/tne/custom/weights/L1 --span_loss L1 --span_epochs 0 --stop_epochs 00 --concurrent_epochs 40 --span_prob 0.8
