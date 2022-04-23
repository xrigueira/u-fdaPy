import os
import numpy as np
import pandas as pd

# Read the csv
File = 'Amonio_full.csv'
fileName, fileExtension = os.path.splitext(File)
df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';', parse_dates=['date'], index_col=['date'])

# Add the needed columns (year, month, day, hour, min, sec)
year = [i for i in df.index.year]
month = [i for i in df.index.month]
day = [i for i in df.index.day]
hour = [i for i in df.index.hour]
minute = [i for i in df.index.minute]
second = [i for i in df.index.second]

df['year'] = year
df['month'] = month
df['day'] = day
df['hour'] = hour
df['minute'] = minute
df['second'] = second

# Save temp file
df.index.name = 'date'
df.to_csv(f'Database/{fileName}_temp.csv', sep=';', encoding='utf-8', index=True, header=['value', 'year', 'month', 'day', 'hour', 'minute', 'second'])

# Add the 31st day to those months with 30
df = pd.read_csv(f'Database/{fileName}_temp.csv', delimiter=';', parse_dates=['date'])
monthShort = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if (x - y) == 29]
print('These are the last indexes of the months with 30 days: ', monthShort)

counter = 0
while monthShort:

    for i in monthShort:

        if df['day'][i-1] == 30:
            
            startDate =  df.iloc[monthShort[0]-1, 0] 
            yearInit, monthInit, dayInit, hourInit, minInit, secInit = startDate.year, startDate.month, startDate.day + 1, 0, 0, 0
            rows = [[f'{yearInit}-{monthInit}-{dayInit} 00:00:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit]]

            for j in range(95):

                minInit += 15

                if minInit > 45:

                    minInit = 0

                    hourInit = hourInit + 1
                
                rows.append([f'{yearInit}-{monthInit}-{dayInit} {hourInit}:{minInit}:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit])

            df2concat = pd.DataFrame(rows, columns=['date', 'value', 'year', 'month', 'day', 'hour', 'minute', 'second'])

            df = pd.concat([df.iloc[:i], df2concat, df.iloc[i:]], ignore_index=True)

            day = df['day']
            
            monthShort = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if (x - y) == 29]

            print('UPDATED indexes of months with 30 days: ', monthShort)
