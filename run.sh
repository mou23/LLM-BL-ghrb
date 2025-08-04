#!/bin/bash
#SBATCH --job-name=llm-bl-ghrb
#SBATCH --output=llm-bl-ghrb.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=5-00:00:00
#SBATCH --mem=128GB
#SBATCH --partition=standard
#SBATCH --account=malek_lab

GROUP="Apache"
GHRB_DIR="../ghrb"
TIME_LOG="pipeline_time_log.txt"

# Prepare logs
echo "Project Execution Time Summary" > "$TIME_LOG"
mkdir -p logs

# Iterate over all XML files (projects)
for xml_file in "$GHRB_DIR"/*.xml
do
    filename=$(basename -- "$xml_file")
    project="${filename%.*}"

    echo "Starting pipeline for project: $project"

    # Start Timer
    start_time=$(date +%s)

    # Run Python and redirect logs
    python LocalizeBug.py "$GROUP" "$project" > "logs/${project}_pipeline.log" 2>&1
    exit_code=$?

    # End Timer
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    formatted_time=$(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60)))

    # Log time
    echo "$project: $formatted_time" >> "$TIME_LOG"

    if [ $exit_code -ne 0 ]; then
        echo "ERROR in project: $project. Exiting."
        echo "Check logs/${project}_pipeline.log for details."
        exit 1  # Stop script on error
    fi

    echo "Finished $project in $formatted_time"

done

echo "====================================="
echo "Pipeline Completed Successfully."
echo "Time summary logged in $TIME_LOG"
echo "====================================="
