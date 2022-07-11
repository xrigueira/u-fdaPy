"""THIS IS AN OLD FILE I NEEDED TO DEVELOP normalizer.py.
It can be deleted when I am 100% sure I won't need it anymore."""

import os
import numpy as np
import pandas as pd
from datetime import datetime


# open the file
file = 'constitucion2.csv'
fileName, fileExtension = os.path.splitext(file)

# set up the date parser
dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y')

# create a data frame and get data from column 'day'
df = pd.read_csv(f'DataBase/{file}', delimiter=';', parse_dates=['date'], date_parser=dateparse, na_values=['NULL'])
cols = list(df.columns.values.tolist())

# Get the index of those months which have 30 days
days = df[cols[3]]
monthShort = [i + 1 for (x, y, i) in zip(days, days[1:], range(len(days))) if (x - y) == 29]
print('Los meses de 30 días están en: ', monthShort)

while monthShort:
    
    for i in monthShort:
    
        if df['day'][i-1] == 30: # insert new row if a month ends in 30 and not 31
        
            if df['month'][i-1] < 10: # this loop is not good coding practices, it just adds a 0 when the month is smaller than 10. It can be clearly improved.
            
                newRow = [str(df['year'][i-1])+'-'+'0'+str(df['month'][i-1])+'-'+str(31)+' 00:00:00', df['year'][i-1], df['month'][i-1], 31, 
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] # OJO aquí, hay que ajustar el número de np.nan (columnas) al que tenga la base de datos
        
                newDf = pd.DataFrame([newRow], columns=cols)
        
                df = pd.concat([df.iloc[:i], newDf, df.iloc[i:]], ignore_index=True)
            
                days = df[cols[3]]
                monthShort = [i + 1 for (x, y, i) in zip(days, days[1:], range(len(days))) if (x - y) == 29]
            
                print('small month')
                print('Los meses de 30 días están en: ', monthShort)

            
            elif df['month'][i-1] >= 10:
            
                newRow = [str(df['year'][i-1])+'-'+str(df['month'][i-1])+'-'+str(31)+' 00:00:00', df['year'][i-1], df['month'][i-1], 31, 
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            
                newDf = pd.DataFrame([newRow], columns=cols)
            
                df = pd.concat([df.iloc[:i], newDf, df.iloc[i:]], ignore_index=True)
            
                days = df[cols[3]]
                monthShort = [i + 1 for (x, y, i) in zip(days, days[1:], range(len(days))) if (x - y) == 29]
            
                print('big month')
                print('Los meses de 30 días están en: ', monthShort)

# Get the index of leaps
days = df[cols[3]]
leaps = [i + 1 for (x, y, i) in zip(days, days[1:], range(len(days))) if ((x - y) == 28 and df['month'][i-1] == 2)]
print('Los febreros están en: ', leaps)

while leaps:
    
    for i in leaps:
            
        if df['day'][i-1] == 29: # insert a row if a month ends in 29th
            
            if df['month'][i-1] < 10:
                
                newRow30 = [str(df['year'][i-1])+'-'+'0'+str(df['month'][i-1])+'-'+str(30)+' 00:00:00', df['year'][i-1], df['month'][i-1], 30, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                newRow31 = [str(df['year'][i-1])+'-'+'0'+str(df['month'][i-1])+'-'+str(31)+' 00:00:00', df['year'][i-1], df['month'][i-1], 31, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                
                newDf = pd.DataFrame([newRow30, newRow31], columns=cols)
                
                df = pd.concat([df.iloc[:i], newDf, df.iloc[i:]], ignore_index=True)
                
                days = df[cols[3]]
                leaps = [i + 1 for (x, y, i) in zip(days, days[1:], range(len(days))) if ((x - y) == 28 and df['month'][i-1] == 2)]
                
                print('small month leap')
                print('Los febreros están en: ', leaps)

            
            elif df['month'][i-1] >= 10:
                
                newRow30 = [str(df['year'][i-1])+'-'+str(df['month'][i-1])+'-'+str(30)+' 00:00:00', df['year'][i-1], df['month'][i-1], 30, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                newRow31 = [str(df['year'][i-1])+'-'+str(df['month'][i-1])+'-'+str(31)+' 00:00:00', df['year'][i-1], df['month'][i-1], 31, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                
                newDf = pd.DataFrame([newRow30, newRow31], columns=cols)
                
                df = pd.concat([df.iloc[:i], newDf, df.iloc[i:]], ignore_index=True)
                
                days = df[cols[3]]
                leaps = [i + 1 for (x, y, i) in zip(days, days[1:], range(len(days))) if ((x - y) == 28 and df['month'][i-1] == 2)]
                
                print('big month leap')
                print('Los febreros están en: ', leaps)


# Get the index of February(s)
days = df[cols[3]]
febs = [i + 1 for (x, y, i) in zip(days, days[1:], range(len(days))) if (x - y) == 27] 
print('Los meses bisiestos están en: ', febs)

while febs:

    for i in febs:
        
        if df['day'][i-1] == 28: # add new row if the month ends in 28
            
            if df['month'][i-1] < 10:
                
                newRow29 = [str(df['year'][i-1])+'-'+'0'+str(df['month'][i-1])+'-'+str(29)+' 00:00:00', df['year'][i-1], df['month'][i-1], 29, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                newRow30 = [str(df['year'][i-1])+'-'+'0'+str(df['month'][i-1])+'-'+str(30)+' 00:00:00', df['year'][i-1], df['month'][i-1], 30, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                newRow31 = [str(df['year'][i-1])+'-'+'0'+str(df['month'][i-1])+'-'+str(31)+' 00:00:00', df['year'][i-1], df['month'][i-1], 31, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

                newDf = pd.DataFrame([newRow29, newRow30, newRow31], columns=cols)
                
                df = pd.concat([df.iloc[:i], newDf, df.iloc[i:]], ignore_index=True)
                
                days = df[cols[3]]
                febs = [i + 1 for (x, y, i) in zip(days, days[1:], range(len(days))) if (x - y) == 27] 
                
                print('small month Feb')
                print('Los meses bisiestos están en: ', febs)
            
            elif df['month'][i-1] >= 10:
                
                newRow29 = [str(df['year'][i-1])+'-'+str(df['month'][i-1])+'-'+str(29)+' 00:00:00', df['year'][i-1], df['month'][i-1], 29, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                newRow30 = [str(df['year'][i-1])+'-'+str(df['month'][i-1])+'-'+str(30)+' 00:00:00', df['year'][i-1], df['month'][i-1], 30, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                newRow31 = [str(df['year'][i-1])+'-'+str(df['month'][i-1])+'-'+str(31)+' 00:00:00', df['year'][i-1], df['month'][i-1], 31, 
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

                newDf = pd.DataFrame([newRow29, newRow30, newRow31], columns=cols)
                
                df = pd.concat([df.iloc[:i], newDf, df.iloc[i:]], ignore_index=True)
                
                days = df[cols[3]]
                febs = [i + 1 for (x, y, i) in zip(days, days[1:], range(len(days))) if (x - y) == 27] 
                
                print('big month Feb')
                print('Los meses bisiestos están en: ', febs)
        
# Interpolate the missing values
# df = (df.interpolate()).round(4)

df = (df.interpolate(method='polynomial', order=1)).round(3)

# Include the information in the database needed to perform the week analysis

# Function to get the index of the firts Monday
def findMonday(Dataframe):
    
    for i in range(len(Dataframe)):
       
       d = datetime(Dataframe['year'][i], Dataframe['month'][i], Dataframe['day'][i])
       
       if d.weekday() == 0:
           print('First Monday index: {} | {}'.format(i, d))
           break
    
    return i

mondayIndex = findMonday(df)

# Add three columsn with the week number, start date and end date, respectively.
weekIndex = []
weekNumber = 0
for i in range(mondayIndex):
    if i < mondayIndex:
        weekIndex.append(0)
        
for i in range(len(df) - mondayIndex):
    
    if i % 7 == 0:
        weekNumber += 1
    weekIndex.append(weekNumber)

df['week'] = weekIndex

startDate = []
for i in range(len((df['week']))):
    
    if i == 0:
        startDate.append('-')  
    
    elif i > 0:
    
        if df['week'][i] != df['week'][i-1]:
            # dateS = (df['year'][i], df['month'][i], df['day'][i])
            year, month, day = 'year', 'month', 'day' # needed to avoid a symtax error
            dateS = f'{df[year][i]} {df[month][i]} {df[day][i]}'
            startDate.append(dateS)
        else:
            startDate.append('-')

df['startDate'] = startDate

endDate = []
for i in range(len((df['week']))):
    
    if i < mondayIndex:
        endDate.append('-')
    
    elif i >= mondayIndex:

        if df['week'][i] != df['week'][i-1]:
            if (i+7) < len(df['week']):
                # dateE = (df['year'][i+6], df['month'][i+6], df['day'][i+6])
                year, month, day = 'year', 'month', 'day' # needed to avoid a syntax error
                dateE = f'{df[year][i+6]} {df[month][i+6]} {df[day][i+6]}'
                endDate.append(dateE)
            else:
                endDate.append('-')
        else:
            endDate.append('-')

df['endDate'] = endDate

# Get the week order within every month
weekOrder = []
weekPosition = 0
for i, e in enumerate(df['week']):
    
    if e == 0:
        weekOrder.append(0)
    
    else:
        if df['week'][i] == df['week'][i-1]:
            weekOrder.append(weekPosition)
        elif df['week'][i] != df['week'][i-1]:
            weekPosition += 1
            if weekPosition == 5:
                weekOrder.append(1)
            else:
                weekOrder.append(weekPosition)
            if weekPosition == 5:
                weekPosition = 1

# print(weekOrder)
df['weekOrder'] = weekOrder

# Update columns
cols = list(df.columns.values.tolist())
print('Variable names: ', cols)

df.to_csv(f'DataBase/{fileName}_pro.csv', sep=';', encoding='utf-8', index=False, header=cols)
