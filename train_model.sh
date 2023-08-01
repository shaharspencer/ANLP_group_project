#!/bin/bash
#SBATCH --gres=gpu:5 #request gpu resources
#SBATCH --nodes=1 # request nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task= #request cpu resources
#SBATCH --time=20:00:00
#SBATCH --job-name=my_gpu_job
#SBATCH --output=train_script_output.out     # Custom output file for standard output
#SBATCH --error=train_script_error.err      # Custom output file for standard error

# activate virtual environment
source /cs/snapless/gabis/shaharspencer/anlp_project_venv/bin/activate

python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate