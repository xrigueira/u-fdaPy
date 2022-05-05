
import os
import numpy as np
import pandas as pd
from datetime import datetime

"""The function checkGaps() look for gaps in the times series and fills them with the missing dates"""

def checkGaps(File):

    # Read the file
    fileName, fileExtension = os.path.splitext(File)
    dateParser = lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M:%S') # It is 4 times faster with the parser
    df = pd.read_csv(f'Database/{fileName}.txt', delimiter=';', parse_dates=['Fecha'], date_parser=dateParser, index_col=['Fecha'])

    # Remove the index duplication
    df = df.loc[~df.index.duplicated(), :]

    # Check for missing dates
    df.index = pd.to_datetime(df.index)
    missingDates = pd.date_range(start='1-1-2019 00:00:00', end='31-12-2021 23:45:00', freq='15min').difference(df.index)

    # Get all the whole date range
    allDates = pd.date_range(start='1-1-2019 00:00:00', end='31-12-2021 23:45:00', freq='15min')

    #  Insert all dates in the db
    df = df.reindex(allDates, fill_value=np.nan)
    df.index.name = 'date'
    
    # Save the db to csv
    df.to_csv(f'Database/{fileName}_full.csv', sep=';', encoding='utf-8', index=True, header=['value'])

