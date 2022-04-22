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

# Add the 31st day to those months with 30
monthShort = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if (x - y) == 29]
print('These are the last indexes of the months with 30 days: ', monthShort)

startDate =  df.iloc[monthShort[0]-1].name # En el while loop el [0] se deja tal y como está porque siempre cogemos el primer valor despues de actualizar la lista
yearInit, monthInit, dayInit, hourInit, minInit, secInit = startDate.year, startDate.month, startDate.day + 1, 0, 0, 0
rows = [[f'{yearInit}-{monthInit}-{dayInit} 00:00:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit]]

for i in range(95):

    minInit += 15

    if minInit > 45:

        minInit = 0

        hourInit = hourInit + 1
    
    rows.append([f'{yearInit}-{monthInit}-{dayInit} {hourInit}:{minInit}:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit])

# Meter desde la fila 29 hasta aquí en el while loop
# https://www.adamsmith.haus/python/answers/how-to-build-a-pandas-dataframe-with-a-for-loop-in-python

# while monthShort:

#     for i in monthShort:

#         if df['day'][i-1] == 30:

#             df2concat = pd.DataFrame()

#             df = pd.concat([df.iloc[:i], df2concat, df.iloc[i:]], ignore_index=True)

#             day = df['day']

#             monthShort = [i + 1 for (x, y, i) in zip(day, day[1:], range(len(day))) if (x - y) == 29]

#             print('UPDATED indexes of months with 30 days: ', monthShort)

# minute = 00
# hour = 00

# for i in range(288):
    
#     time = f'{hour}:{minute}'
    
#     print(time)
    
#     if minute == 55:
#         hour = hour + 1
#         minute = 0
#     else:
#         minute = minute + 5

