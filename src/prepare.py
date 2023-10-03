import pandas as pd
import sys
import random
import os

os.makedirs(os.path.join('data'), exist_ok=True)

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython prepare.py data-file\n")
    sys.exit(1)


data_path = sys.argv[1]

print(data_path)

df = pd.read_csv(data_path)

all_features = df.columns

# Let's drop some features
names = [feat for feat in all_features if "net_name" in feat] # excluded for privacy reasons
useless = ["info_gew","info_resul","interviewtime","id","date"] # features that we expect are uninformative
drop_list = names + useless 

# Remove the questionnaire about agricultural practices until I can better understand it
practice_list = ["legum","conc","add","lact","breed","covman","comp","drag","cov","plow","solar","biog","ecodr"]
for feat in all_features:
    if any(x in feat for x in practice_list):
        drop_list.append(feat)


df = df.drop(columns=drop_list)

# Convert non-numeric features to numeric
non_numeric = list(df.select_dtypes(include=['O']).columns)
for col in non_numeric:
    codes,uniques=pd.factorize(df[col])
    df[col] = codes

df.to_csv('data/data_prepared.csv',index = False)

