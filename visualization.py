import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
train = pd.read_csv('Dataset/train_all.csv', index_col='id')
test = pd.read_csv('Dataset/test_all.csv', index_col='id')

# Visualize the distribution for each feature
for i in range(len(train.columns)):

    sns.distplot(train.iloc[:, i])
    plt.show()

# Find the correlations between each pair of features with pairplot
sns.pairplot(train)
plt.show()

# Find the correlations between each pair of features with heatmap
sns.heatmap(train.corr())
plt.show()

# Plot ann output with histogram
ann_output = pd.read_csv('ann_output.csv', index_col=['id'])
sns.distplot(ann_output['class'])
plt.show()
