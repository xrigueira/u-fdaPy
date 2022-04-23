
import os
import pandas as pd
import numpy as np

File = 'Amonio_nor.csv'
fileName, fileExtension = os.path.splitext(File)
df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')

years = list(dict.fromkeys(df['year'].tolist()))
months = list(dict.fromkeys(df['month'].tolist()))
months.sort()

indexInit, indexEnd = [], []
for i in years:

    df = df.loc[df['year'] == i]
    
    for j in months:
        
        df = df.loc[df['month'] == j]
    
        numNaN = df['value'].isnull().sum()
        
        # Get the first and last index of those months with too many empty values (NaN in this case)
        if numNaN >= 480:
            indexInit.append(df.index[0])
            indexEnd.append(df.index[-1])

        df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
        
        if j == 12:
            df = df.loc[df['year'] == (i+1)]
        else:
            df = df.loc[df['year'] == i]


# Delete those parts of the data frame between the appended indices
df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')

toDrop = []
for i,j in zip(indexInit, indexEnd):

    toDrop.append(str(i)+':'+str(j))

print(toDrop)

# df = df.drop(np.r_[0:2975, 2976:5951, 17856:20831]) # Now I need to make this process automatic somehow
# df = df.drop(df.index[0:2975], inplace=False)

# print(len(df))
