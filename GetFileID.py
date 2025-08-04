import json

with open('expresults/dubbo/11561/codenames.json', 'r') as file:
    file_list = json.load(file)

for idx, file_path in enumerate(file_list, start=0):
    print(f"{idx}: {file_path}")
