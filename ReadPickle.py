# import pickle

# with open('expresults/aspectj/reports.pkl', 'rb') as file:
#     data = pickle.load(file)

# print(data)

import pandas as pd

df = pd.read_pickle('expresults/Apktool/3113/reports.pkl')

df.to_csv('expresults/Apktool/3113/reports.csv', index=False)


# import pickle

# with open('expresults/dubbo/11561/candidate.pkl', 'rb') as f:
#     data = pickle.load(f)

# for key, value in data.items():
#     print(f"{key}: {value}")
