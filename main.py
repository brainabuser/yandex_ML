import pandas as pnd

import re

data = pnd.read_csv('titanic.csv')

# print(data.to_string())

# Amount of both male and female Titanic passengers
file1 = r'answers\1.txt'
output1 = data['Sex'].value_counts()
with open(file1, 'w') as file1_obj:
    file1_obj.write(str(output1['male']) + ' ' + str(output1['female']))

# Percentage of survived Titanic passengers
file2 = r'answers\2.txt'
output2 = data['Survived'].value_counts(normalize=True)
with open(file2, 'w') as file1_obj:
    file1_obj.write(str(round(output2[1] * 100, 2)))

# Percentage of 1st class Titanic passengers
file3 = r'answers\3.txt'
output3 = data['Pclass'].value_counts(normalize=True)
with open(file3, 'w') as file1_obj:
    file1_obj.write(str(round(output3[1] * 100, 2)))

# Median and average of the total number of Titanic passengers` age
file4 = r'answers\4.txt'
ages = data['Age']
with open(file4, 'w') as file1_obj:
    file1_obj.write(str(ages.mean()) + ' ' + str(ages.median()))

# Pearson's correlation between siblings and parents/children
file5 = r'answers\5.txt'
with open(file5, 'w') as file1_obj:
    file1_obj.write(str(data['SibSp'].corr(data['Parch'])))

# Most popular female name
female = data['Sex'] == 'female'
women = data[female]

def analize(line):
    # Remove aliases
    # '.+\. '-> ends with dot symbol
    # ' ?\".+\"' -> enclosed in double quotes
    line = re.sub(r'.+\. | ?\".+\"', "", line)
    # ' ?\(\)' -> empty parentheses
    #  ' ?\b[a-zA-Z]\b' -> single(two)-letter words
    line = re.sub(r' ?\(\)| ?\b[a-zA-Z]{1,2}\b', "", line)
    # '\(.+\)' -> maiden names
    seek = re.search(r'\(.+\)', line)
    if seek is not None:
        line = seek.group()
        line = line[1:len(line) - 1]
    names = line.split()
    if len(names) > 1:
        names.pop()
    return names


names = []
for i in women['Name']:
    words = analize(i)
    while len(words) != 0:
        names.append(words.pop())

wordfreq = [names.count(w) for w in names]

NamesDataSet = list(zip(names, wordfreq))
df = pnd.DataFrame(data=NamesDataSet, columns=['Names', 'Freqs'])
df = df.groupby('Names').mean()
df.sort_values("Freqs", inplace=True, ascending=False)

file6 = r'answers\6.txt'
with open(file6, 'w') as file1_obj:
    file1_obj.write(str(df.head(1).index.values[0]))
