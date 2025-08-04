#!/bin/bash
#SBATCH --job-name=llm-bl-ghrb
#SBATCH --output=llm-bl-ghrb.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=5-00:00:00
#SBATCH --mem=128GB
#SBATCH --partition=gpu
#SBATCH --account=malek_lab_gpu
module load anaconda/2024.06 

# Check if project name is passed as argument
if [ -z "$1" ]; then
    echo "Usage: $0 <ProjectName>"
    exit 1
fi

GROUP="Apache"
PROJECT="$1"
GHRB_DIR="../ghrb"
TIME_LOG="pipeline_time_log.txt"

# Prepare logs
echo "Project Execution Time Summary" > "$TIME_LOG"
mkdir -p logs

echo "Starting pipeline for project: $PROJECT"

# Start Timer
start_time=$(date +%s)

# Run Python and redirect logs
python LocalizeBug.py "$GROUP" "$PROJECT" > "logs/${PROJECT}_pipeline.log" 2>&1
exit_code=$?

# End Timer
end_time=$(date +%s)
elapsed=$((end_time - start_time))
formatted_time=$(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60)))

# Log time
echo "$PROJECT: $formatted_time" >> "$TIME_LOG"

if [ $exit_code -ne 0 ]; then
    echo "ERROR in project: $PROJECT. Exiting."
    echo "Check logs/${PROJECT}_pipeline.log for details."
    exit 1  # Stop script on error
fi

echo "Finished $PROJECT in $formatted_time"

echo "====================================="
echo "Pipeline Completed Successfully."
echo "Time summary logged in $TIME_LOG"
echo "====================================="
