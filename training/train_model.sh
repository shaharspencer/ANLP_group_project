#!/bin/bash
#SBATCH --gres=gpu:2 #request gpu resources
#SBATCH --nodes=1 # request nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 #request cpu resources
#SBATCH --time=20:00:00
#SBATCH --job-name=my_gpu_job
#SBATCH --output=train_script_output.out     # Custom output file for standard output
#SBATCH --error=train_script_error.err      # Custom output file for standard error

module load cuda/11.7
# activate virtual environment
source /cs/snapless/gabis/shaharspencer/anlp_project_venv/bin/activate

module load cuda/11.7

python run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file ../data_files/abstracts_introductions_train_split.csv \
    --validation_file ../data_files/abstracts_introductions_validation_split.csv \
    --source_prefix "summarize: " \
    --output_dir trained_t5_small_model \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
     --text_column introduction \
    --summary_column abstract