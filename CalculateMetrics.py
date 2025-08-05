import os
import json

def calculate_accuracy_at_k(bug_data):
    for top in [1,5,10]:
        count = 0
        total_bug = 0
        for bug in bug_data:
            suspicious_files = bug['suspicious_files']
            # length_of_suspicious_files = len(suspicious_files)

            fixed_files = bug['fixed_files']

            # fixed_files = bug['fixed_files'].split('.java')
            # fixed_files = [(file + '.java').strip() for file in fixed_files[:-1]]

            # print(bug['bug_id'], fixed_files)
            for fixed_file in fixed_files:
                if fixed_file in suspicious_files[0:top]:
                    print(bug['bug_id'],fixed_file)
                    count = count + 1
                    break
            total_bug = total_bug + 1
        print('accuracy@', top, count, total_bug, (count*100/total_bug))


def calculate_mean_reciprocal_rank_at_k(bug_data):
    for top in [10]:
        total_bug = 0
        inverse_rank = 0
        for bug in bug_data:
            suspicious_files = bug['suspicious_files']
            length_of_suspicious_files = len(suspicious_files)
            fixed_files = bug['fixed_files']

            # fixed_files = bug['fixed_files'].split('.java')
            # fixed_files = [(file + '.java').strip() for file in fixed_files[:-1]]
            # print("ID ",item['bug_id'])
            # print(suspicious_files)
            # print("length_of_suspicious_files",length_of_suspicious_files)
            minimum_length = min(top,length_of_suspicious_files)
            for i in range(minimum_length):
                if(suspicious_files[i] in fixed_files):
                    # print('first rank', item['bug_id'], i+1, suspicious_files[i])
                    inverse_rank = inverse_rank + (1/(i+1))
                    break
            total_bug = total_bug + 1
        if inverse_rank == 0:
            print("MRR@", top, 0)
        else:
            print("MRR@", top, (1/total_bug)*inverse_rank)
           
     
def calculate_mean_average_precision_at_k(bug_data):
    for top in [10]:
        total_bug = 0
        total_average_precision = 0
        for bug in bug_data:
            average_precision = 0
            precision = 0
            suspicious_files = bug['suspicious_files']
            length_of_suspicious_files = len(suspicious_files)
            fixed_files = bug['fixed_files']

            # fixed_files = bug['fixed_files'].split('.java')
            # fixed_files = [(file + '.java').strip() for file in fixed_files[:-1]]
            number_of_relevant_files = 0
            minimum_length = min(top,length_of_suspicious_files)
            for i in range(minimum_length):
                # print("i",i)
                if(suspicious_files[i] in fixed_files):
                    # print(item['bug_id'],suspicious_files[i], " relevant")
                    number_of_relevant_files = number_of_relevant_files + 1                        
                    precision = precision + (number_of_relevant_files/(i+1))
                # print("precision ", precision)
            average_precision = precision/len(fixed_files)
            # print("average_precision" ,average_precision, len(fixed_files))
            total_average_precision = total_average_precision + average_precision
            total_bug = total_bug + 1
        mean_average_precision = total_average_precision/total_bug
        print("MAP@", top, mean_average_precision)


base_dir = "expresults"
bug_data = []

# Iterate through each project directory inside expresults
for project_name in os.listdir(base_dir):
    project_path = os.path.join(base_dir, project_name)
    if not os.path.isdir(project_path):
        continue  # Skip files, only process directories

    # Find all bug-id directories (e.g., 7048)
    bug_ids = [d for d in os.listdir(project_path) if d.isdigit()]
    
    for bug_id in bug_ids:
        bug_dir = os.path.join(project_path, bug_id)

        # Locate the *_prompts.json file
        prompts_file = None
        for file in os.listdir(bug_dir):
            if file.endswith("_prompts.json"):
                prompts_file = os.path.join(bug_dir, file)
                break
        
        if not prompts_file:
            print(f"No prompts.json found for {project_name}/{bug_id}")
            continue
        
        # Load prompt data
        with open(prompts_file, 'r') as f:
            prompt_data = json.load(f)
        
        bug_entry = prompt_data.get(bug_id)
        if not bug_entry:
            print(f"No entry for bug_id {bug_id} in prompts.json")
            continue
        
        file_names = bug_entry.get("names", [])
        fixed_files = bug_entry.get("files", [])

        # Find the corresponding all_sen_9_*/{bug_id}_llm.txt file
        llm_file_path = None
        for d in os.listdir(project_path):
            if d.startswith("all_sen_9_"):
                llm_folder = os.path.join(project_path, d)
                llm_file_candidate = os.path.join(llm_folder, f"{bug_id}_llm.txt")
                if os.path.isfile(llm_file_candidate):
                    llm_file_path = llm_file_candidate
                    break
        
        if not llm_file_path:
            print(f"No LLM file found for {project_name}/{bug_id}")
            continue
        
        with open(llm_file_path, 'r') as f:
            llm_data = json.load(f)
            llm_indices = llm_data[0]  # e.g., [3, 4, 1, 0, ...]

        suspicious_files = [file_names[i] for i in llm_indices]

        bug_data_entry = {
            'bug_id': f"{project_name}-{bug_id}",
            'fixed_files': fixed_files,
            'suspicious_files': suspicious_files
        }
        print(bug_data_entry)
        bug_data.append(bug_data_entry)

        
        calculate_accuracy_at_k(bug_data)
        calculate_mean_reciprocal_rank_at_k(bug_data)
        calculate_mean_average_precision_at_k(bug_data)
