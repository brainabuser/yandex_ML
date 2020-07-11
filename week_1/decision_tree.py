import pandas as pnd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

data = pnd.read_csv('titanic.csv')
data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna()
Survivors = data.pop('Survived')

# Replace: male -> 0, female -> 1
data.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)

print(data)

X = np.array(data)
y = np.array(Survivors)
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = clf.feature_importances_

sorted_indices = np.argsort(importances)

r_indices = sorted_indices[::-1]

line = ''
for i in data.iloc[:, r_indices[:2]].columns.values:
    line += i + ' '

file = r'answers\7.txt'
with open(file, 'w') as file_obj:
    file_obj.write(line.rstrip())
