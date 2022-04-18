#! /bin/sh
#SBATCH --job-name=curriculum_start_span
#SBATCH --output=slurm_configs/slurm_out/curriculum_start_span/curriculum_start_span.out
#SBATCH --error=slurm_configs/slurm_out/curriculum_start_span/curriculum_start_span.err
#SBATCH --partition=studentbatch
#SBATCH --time=720
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

python /home/joberant/NLP_2122/idangrosbard/TNE/tne/custom/train_gpt_prompt.py /home/joberant/NLP_2122/idangrosbard/TNE/data/train.jsonl /home/joberant/NLP_2122/idangrosbard/TNE/data/dev.jsonl /home/joberant/NLP_2122/idangrosbard/TNE/tne/custom/weights/curriculum_start_span/ --span_loss L2 --span_epochs 20 --stop_epochs 0 --concurrent_epochs 20 --span_prob 0.8
