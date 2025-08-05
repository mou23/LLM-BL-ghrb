import os
import json


directory = "expresults/"

empty_count = 0
non_empty_count = 0
all_values = []

for filename in os.listdir(directory):
    if filename.endswith("_all_sen_9_filter.json"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
            if not data:  # empty list
                empty_count += 1
            else:
                non_empty_count += 1
                all_values.extend(data)

# Print results
print(f"Number of bugs without correct candidates: {empty_count}")
print(f"Number of bugs with correct candidates: {non_empty_count}")
print(f"\nBug IDs with correct candidates: {all_values}")
