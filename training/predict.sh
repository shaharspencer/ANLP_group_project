#!/bin/bash
#SBATCH --gres=gpu:1,vmem:24G #request gpu resources
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH --time=01:00:00
#SBATCH --job-name=my_gpu_job
#SBATCH --output=train_script_output_t5_large.out  # Custom output file for standard output
#SBATCH --error=train_script_error_t5_large.err    # Custom output file for standard error

module load cuda/11.7
# activate virtual environment
source /cs/snapless/gabis/shaharspencer/anlp_project_venv/bin/activate

module load cuda/11.7

python /cs/snapless/gabis/shaharspencer/ANLP_group_project/prepare.py

trained_t5_large_model_routing_protocols
python /cs/snapless/gabis/shaharspencer/ANLP_group_project/training/run_summarization.py \
    --config_name /cs/snapless/gabis/shaharspencer/ANLP_group_project/model_outputs/trained_t5_large_model_routing_protocols/config.json \
    --model_name_or_path /cs/snapless/gabis/shaharspencer/ANLP_group_project/model_outputs/trained_t5_large_model_routing_protocols \
    --do_predict \
    --test_file /cs/snapless/gabis/shaharspencer/ANLP_group_project/data_files/neural_network_verification/abstracts_introductions_test_split.csv \
    --source_prefix "summarize: " \
    --output_dir /cs/snapless/gabis/shaharspencer/ANLP_group_project/model_outputs/trained_t5_large_model_routing_protocols \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --predict_with_generate \
    --text_column introduction \
    --summary_column abstract