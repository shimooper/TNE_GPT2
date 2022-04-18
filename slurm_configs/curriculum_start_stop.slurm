#! /bin/sh
#SBATCH --job-name=curriculum_start_stop
#SBATCH --output=slurm_configs/slurm_out/curriculum_start_stop/curriculum_start_stop.out
#SBATCH --error=slurm_configs/slurm_out/curriculum_start_stop/curriculum_start_stop.err
#SBATCH --partition=studentbatch
#SBATCH --time=720
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python /home/joberant/NLP_2122/idangrosbard/TNE/tne/custom/train_gpt_prompt.py /home/joberant/NLP_2122/idangrosbard/TNE/data/train.jsonl /home/joberant/NLP_2122/idangrosbard/TNE/data/dev.jsonl /home/joberant/NLP_2122/idangrosbard/TNE/tne/custom/weights/curriculum_start_stop/ --span_loss L2 --span_epochs 0 --stop_epochs 20 --concurrent_epochs 20 --span_prob 0.8
