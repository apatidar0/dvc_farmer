import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import joblib
import tarfile
import zipfile
import os , sys
from sklearn.metrics import confusion_matrix, classification_report


os.makedirs(os.path.join('artifact'), exist_ok=True)

input_path = sys.argv[1]
print("input path",input_path)

df = pd.read_csv(input_path)

#### Get features ready to model! 
y = df.pop("cons_general").to_numpy()
y[y< 4] = 0
y[y>= 4] = 1

X = df.to_numpy()
X = preprocessing.scale(X) # Is standard
# Impute NaNs

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)

# Linear model
clf = LogisticRegression()
yhat = cross_val_predict(clf, X, y, cv=5)

acc = np.mean(yhat==y)
tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp + fn)

# Now print to file
with open("artifact/metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc, "specificity": specificity, "sensitivity":sensitivity}, outfile)

# Let's visualize within several slices of the dataset
score = yhat == y
score_int = [int(s) for s in score]
df['pred_accuracy'] = score_int

# Bar plot by region

sns.set_color_codes("dark")
ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette = "Greens_d")
ax.set(xlabel="Region", ylabel = "Model accuracy")
plt.savefig("artifact/by_region.png",dpi=80)


# Save the best model using joblib
joblib.dump(clf, 'artifact/Logistic.joblib')


# Create a tar.gz file
with tarfile.open('artifact/Logistic_model.tar.gz', 'w:gz') as tar:
    tar.add('artifact/Logistic.joblib')


# Calculate confusion matrix
confusion = confusion_matrix(y, yhat)

# Extract values from confusion matrix
tn, fp, fn, tp = confusion.ravel()

# Calculate specificity and sensitivity
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(6, 4))  # Set dimensions here (width, height)
sns.set(font_scale=1.2)
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Save the heatmap as confusion_matrix.png
plt.savefig("artifact/confusion_matrix.png", dpi=80)

# Save a table form of the classification report
classification_report_df = pd.DataFrame(classification_report(y, yhat, output_dict=True)).transpose()
classification_report_df.to_csv('artifact/classification_report.csv', index=True)


