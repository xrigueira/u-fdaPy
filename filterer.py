
import os
import pandas as pd

"""This function deletes those time spans with too many empty values,
and iterates on the rest"""

def filterer(File, timeframe):
        
    fileName, fileExtension = os.path.splitext(File)
    df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')

    years = list(dict.fromkeys(df['year'].tolist()))

    months = list(dict.fromkeys(df['month'].tolist()))
    months.sort()

    weeks = list(dict.fromkeys(df['week'].tolist()))
    weekOrder = list(dict.fromkeys(df['weekOrder'].tolist()))

    startDate = list(df['startDate'])
    endDate = list(df['endDate'])

    days = list(dict.fromkeys(df['day'].tolist()))


    if timeframe == 'a':
        
        indexInit, indexEnd = [], []
        for i in years:

            df = df.loc[df['year'] == i]
            
            for j in months:
                
                df = df.loc[df['month'] == j]

                # Get total number of NaN and the max consecutive NaNs
                numNaN = df['value'].isnull().sum()
                consecNaN = max(df['value'].isnull().astype(int).groupby(df['value'].notnull().astype(int).cumsum()).sum())
                
                # Get the first and last index of those months with too many empty (or consecutive) values (NaN in this case)
                if numNaN >= 480 or consecNaN >= 192:
                    indexInit.append(df.index[0])
                    indexEnd.append(df.index[-1])

                df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
                
                if j == 12:
                    df = df.loc[df['year'] == (i+1)]
                else:
                    df = df.loc[df['year'] == i]


        # Delete those parts of the data frame between the appended indices
        df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')

        counter = 0
        lenMonth = 2976
        for i,j in zip(indexInit, indexEnd):

            df = df.drop(df.index[int(i-counter*lenMonth):int(j-counter*lenMonth+1)], inplace=False)
            counter += 1
        
        # Interpolate the remaining empty values
        df = (df.interpolate(method='polynomial', order=1)).round(2)
        
        # Save the data frame 
        cols = list(df.columns.values.tolist())
        df.to_csv(f'Database/{fileName[0:-4]}_pro.csv', sep=';', encoding='utf-8', index=False, header=cols)

    elif timeframe == 'b':
        
        weeks = [i for i in weeks if i != 0] # Remove the 0

        indexInit, indexEnd = [], []
        for i in weeks:
            
            df = df.loc[df['week'] == i]

            # Get total number of NaN and the max consecutive NaNs
            numNaN = df['value'].isnull().sum()
            consecNaN = max(df['value'].isnull().astype(int).groupby(df['value'].notnull().astype(int).cumsum()).sum())
            # Get the first and last index of those weeks with too many empty (or consecutive) values (NaN in this case)
            if numNaN >= 192 or consecNaN >= 24:
                indexInit.append(df.index[0])
                indexEnd.append(df.index[-1])

            df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
            
        # Delete those parts of the data frame between the appended indices
        df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
        
        counter = 0
        lenWeek = 672
        for i, j in zip(indexInit, indexEnd):
            
            df = df.drop(df.index[int(i-counter*lenWeek):int(j-counter*lenWeek+1)], inplace=False)
            counter += 1
        
        # Interpolate the remaining empty values
        df = (df.interpolate(method='polynomial', order=1)).round(2)

        # Save the data frame
        cols = list(df.columns.values.tolist())
        df.to_csv(f'Database/{fileName[0:-4]}_pro.csv', sep=';', encoding='utf-8', index=False, header=cols)
        
    elif timeframe == 'c':

        indexInit, indexEnd = [], []
        for i in years:

            df = df.loc[df['year'] == i]

            for j in months:

                df = df.loc[df['month'] == j]

                for k in days:

                    df = df.loc[df['day'] == k]

                    # Get total number of NaN and the max consecutive NaNs
                    numNaN = df['value'].isnull().sum()
                    consecNaN = max(df['value'].isnull().astype(int).groupby(df['value'].notnull().astype(int).cumsum()).sum())
                    # Get the first and last index of those days with too many empty (or consecutive) values (NaN in this case)
                    if numNaN >= 20 or consecNaN >= 8:
                        indexInit.append(df.index[0])
                        indexEnd.append(df.index[-1])

                    df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')

                    if j == 12 and k == 31:
                        df = df.loc[df['year'] == (i+1)]
                        df = df.loc[df['month'] == 1]
                    
                    elif j <= 12:
                        df = df.loc[df['year'] == i]
                        
                        if k < 31:
                            df = df.loc[df['month'] == j]
                        
                        elif k == 31:
                            df = df.loc[df['month'] == (j+1)]

        # Delete those parts of the data frame between the appended indices
        df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')

        counter = 0
        lenDay = 96
        for i,j in zip(indexInit, indexEnd):

            df = df.drop(df.index[int(i-counter*lenDay):int(j-counter*lenDay+1)], inplace=False)
            counter += 1

        # Interpolate the remaining empty values
        df = (df.interpolate(method='polynomial', order=1)).round(2)

        # Save the data frame
        cols = list(df.columns.values.tolist())
        df.to_csv(f'Database/{fileName[0:-4]}_pro.csv', sep=';', encoding='utf-8', index=False, header=cols)
