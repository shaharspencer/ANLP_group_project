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


# IMPORTANT! If not Shahar's user, change to your user only here!
# Example: cd /cs/snapless/gabis/nive/ANLP_group_project/
cd /cs/snapless/gabis/shaharspencer/ANLP_group_project/

# git pull

python prepare.py

python model_scripts/training/run_summarization.py \
    --model_name_or_path model_outputs/trained_t5_large_model_GENERIC \
    --config_name model_outputs/trained_t5_large_model_GENERIC/config.json \
    --training_dataset_name GENERIC \
    --prediction_dataset_name GENERIC \
    --do_predict \
    --test_file data_files/GENERIC/abstracts_introductions_test_split.csv \
    --source_prefix "summarize: " \
    --output_dir model_outputs/trained_t5_large_model_GENERIC \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --predict_with_generate \
    --text_column introduction \
    --summary_column abstract