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

python prepare.py

python training/run_summarization.py \
    --model_name_or_path /cs/snapless/gabis/shaharspencer/ANLP_group_project/model_outputs/trained_t5_small_model_neural_net \
    --config_name /cs/snapless/gabis/shaharspencer/ANLP_group_project/model_outputs/trained_t5_small_model_neural_net/config.json\
    --do_predict \
    --test_file /cs/snapless/gabis/shaharspencer/ANLP_group_project/data_files/neural_net/abstracts_introductions_test_split.csv \
    --source_prefix "summarize: " \
    --output_dir /cs/snapless/gabis/shaharspencer/ANLP_group_project/model_outputs/trained_t5_small_model_neural_net \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
    --text_column introduction \
    --summary_column abstract