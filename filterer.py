
import os
import pandas as pd

"""This function deletes those time spans with too many empty values,
and iterates on the rest"""

def filterer(File, span):
        
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


    if span == 'a':
        
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

        counter = 0
        lenMonth = 2976
        for i,j in zip(indexInit, indexEnd):

            df = df.drop(df.index[int(i-counter*lenMonth):int(j-counter*lenMonth+1)], inplace=False)
            counter += 1
        
        # Interpolate the remaining empty values
        df = (df.interpolate(method='polynomial', order=1)).round(2)
        
        # Save the data frame 
        cols = list(df.columns.values.tolist())
        df.to_csv(f'Database/{fileName[0:-5]}_pro.csv', sep=';', encoding='utf-8', index=False, header=cols)

    elif span == 'b':
        
        indexInit, indexEnd = []
        for i in weeks:
            
            df = df.loc[df['week'] == i]
            
            numNaN = df['value'].isnull().sum()
            
            # Get the first and last index of those weeks with too many empty values (NaN in this case)
            if numNaN >= 194:
                indexInit.append(df.index[0])
                indexEnd.append(df.index[-1])

            df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
            
        # Delete those parts of the data frma between the appended indices
        df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
        
        counter = 0
        lenWeek = 672
        for i, j in zip(indexInit, indexEnd):
            pass
        # TODO: Continue coding here
        
    elif span == 'c':
        pass
