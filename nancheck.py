import os
import pandas as pd
import numpy as np

# Read the database
File = 'Oxigeno disuelto'

fileName, fileExtension = os.path.splitext(File)
df = pd.read_csv(f'Database/{fileName}_pro.csv', delimiter=';', parse_dates=['date'], index_col=['date'])

# Check for NaN in a single df column
print('Are there any NaNs in the column?', df['value'].isnull().values.any())

# Count the NaN under a single data frame colum
print('Total number of NaNs in the df', df['value'].isnull().sum())

# Check for NaN under an entire data frame
print('Are there any NaNs in the df?', df.isnull().values.any())

# Count the NaN under an entire data frame
print('Total number of NaN in the df ', df.isnull().sum().sum())

print('Index of the NaNs', df.index[df.isnull().any(1)])


# Get the index where there is an inf
# print('Index where there is an inf', df.index[np.isinf(df).any(1)])