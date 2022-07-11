import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

"""This file gets the correlation matrix"""

# Load the data into a pandas df
File = 'data_pro.csv'
df = pd.read_csv(f'Database/{File}', delimiter=';', parse_dates=['date']) # Got rid off the date column as index because it is needless

# Select the desired coluns
cols = ['ammonium', 'conductivity', 'nitrates', 'oxygen', 'pH', 'temperature', 'turbidity', 'flow', 'pluviometry']

df = df[cols]

corr_matrix = df.corr().round(4)
print(corr_matrix)

sns.heatmap(corr_matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()