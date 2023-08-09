#!/bin/bash
#SBATCH --gres=gpu:1,vmem:24G #request gpu resources
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --time=20:00:00
#SBATCH --job-name=my_gpu_job
#SBATCH --output=train_script_output_t5_large.out  # Custom output file for standard output
#SBATCH --error=train_script_error_t5_large.err    # Custom output file for standard error

module load cuda/11.7
# activate virtual environment
source /cs/snapless/gabis/shaharspencer/anlp_project_venv/bin/activate

module load cuda/11.7

python /cs/snapless/gabis/shaharspencer/ANLP_group_project/prepare.py

python /cs/snapless/gabis/shaharspencer/ANLP_group_project/training/run_summarization.py \
    --model_name_or_path t5-large \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file /cs/snapless/gabis/shaharspencer/ANLP_group_project/data_files/routing_protocols/abstracts_introductions_train_split.csv \
    --validation_file /cs/snapless/gabis/shaharspencer/ANLP_group_project/data_files/routing_protocols/abstracts_introductions_validation_split.csv \
    --test_file /cs/snapless/gabis/shaharspencer/ANLP_group_project/data_files/routing_protocols/abstracts_introductions_test_split.csv \
    --source_prefix "summarize: " \
    --output_dir /cs/snapless/gabis/shaharspencer/ANLP_group_project/model_outputs/trained_t5_large_model_routing_protocols \
    --overwrite_output_dir=True \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --predict_with_generate \
    --text_column introduction \
    --summary_column abstract