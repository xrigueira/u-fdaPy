"""Nelson Rules

Rule 1: one point is more than 3 standars deviations from the mean.

Rule 2: Nine (or more) points in a row are on the same side of the mean.

Rule 3: Six (or more) points in a row are continually increasing (or decreasing).

Rule 4: Fourteen (or more) points in a row alternate in direction, increasing then 
decreasing.

Rule 5: Two (or three) out of three points in a row are more than 2 standard deviations 
from the mean in the same direction.

Rule 6: Four (or five) out of five points in a row are more than 1 standard deviation 
from the mean in the same direction.

Rule 7: Fifteen points in a row are all within 1 standard deviation of the mean on 
either side of the mean.

Rule 8: Eight points in a row exist, but none within 1 standard deviation of the mean, 
and the points are in both directions from the mean.
"""

import os
import math
import random
import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# File selection
file = 'argentina2_pro.csv'
fileName, fileExtension = os.path.splitext(f'DataBase\{file}')

# Read the database
df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])
cols = list(df.columns.values.tolist())
dates = list(df.index)
print('Variable names: ', cols[3:-4])

# Start user questionnaire to know what to plot:
varName = str(input('Insert the name of the variable as it is in the database: '))
# varName = 'no2'
timeFrame = input('Years (a), months (b), weeks (c), or days (d): ')
# timeFrame = 'c'
# Crop the database to the time frame desired
if timeFrame == 'a':
    span = input('Use a range of years (Y/n): ')
    
    if span == 'Y':
        print('----- Use the same year if just want to work with one-----')
        
        startDate = int(input('Insert the first year: '))
        endDate = int(input('Insert the last year: '))
        mask = (df['year'] >= startDate) & (df['year'] <= endDate)
        
        df = df.loc[mask]
        
    else:
        pass

elif timeFrame == 'b':
    span = input('All months (a), all months in one or several given years (b), a specific month in every year (c), several months in every year (d) several months in several years (e) or a range of months (f): ')
    # span = 'a'
    if span == 'a':
        monthEnd = 0
    
    elif span == 'b':
        print('----- Use the same year if just want to work with one-----')
        
        startDate = int(input('Insert the first year: '))
        endDate = int(input('Insert the last year: '))
        mask = (df['year'] >= startDate) & (df['year'] <= endDate)
        
        df = df.loc[mask]

        monthEnd = 0
        
    elif span == 'c':
        numberMonths = int(input('Enter the month number: '))
        df = df.loc[df['month'] == numberMonths]

        monthEnd = 0
    
    elif span == 'd':
        monthStart, monthEnd = int(input('Enter the starting month: ')), int(input('Enter the ending month: '))
        df = df[(df['month'] >= monthStart) & (df['month'] <= monthEnd)]
    
    elif span == 'e':
        yearStart, yearEnd = int(input('Enter the starting year: ')), int(input('Enter the ending year: '))
        monthStart, monthEnd = int(input('Enter the starting month: ')), int(input('Enter the ending month: '))
        df = df[(df['year'] >= yearStart) & (df['year'] <= yearEnd)]
        df = df[(df['month'] >= monthStart) & (df['month'] <= monthEnd)]

    elif span == 'f':
        startDate = str(input('Enter the starting date (YYYY-MM-DD hh:mm:ss): '))
        endDate = str(input('Enter the ending date (YYYY-MM-DD hh:mm:ss): '))

        df = df.loc[startDate:endDate]

        monthEnd = 0

elif timeFrame == 'c':
    span = input('All weeks (a), 1st/2nd/3rd/4th week of each month (b), range of weeks in several or all years (c) or range of weeks (d): ')

    if span == 'a':
        df = df.loc[df['week'] != 0]
        
    elif span == 'b':
        weekNumber = int(input('Enter the week number: '))
        df = df.loc[df['weekOrder'] == weekNumber]
    
    elif span == 'c':
        yearBeginOri, monthBegin, dayBegin = input('Enter the first year desired: '), input('Enter the first month desired: '), input('Enter the first day desired: ')
        numberYears, numberMonths = int(input('Enter number of years to analyze: ')), int(input('Enter the number of months to analyze: '))

    elif span == 'd':
        yearBegin, monthBegin, dayBegin = input('Enter the first year desired: '), input('Enter the first month desired: '), input('Enter the first day desired: ')
        yearEnd, monthEnd, dayEnd = input('Enter the last year desired: '), input('Enter the last month desired: '), input('Enter the last day desired: ')
        
        startDate = list(df['startDate'])
        endDate = list(df['endDate'])
        
        # Start variables
        daysAvailableS = []
        daysAvailableE = []
    
        # Clean startDate and endDate
        startDate = [i for i in startDate if i != '-']
        endDate = [i for i in endDate if i != '-']
        
        # Get the closes starting date to the user input
        for i in startDate:
            
            if i[0:4] == yearBegin and int(i[5:7]) == int(monthBegin):
                daysAvailableS.append(int(i[-2:]))
            
        closestDayS = min(daysAvailableS, key=lambda x:abs(x-int(dayBegin)))
        closestStartDate = f'{int(yearBegin)} {int(monthBegin)} {closestDayS}'
        print(closestStartDate)
        # Get the closes end date to the users input
        for i in endDate:
            
            if i[0:4] == yearEnd and int(i[5:7])== int(monthEnd):
                daysAvailableE.append(int(i[-2:]))
        
        closestDayE = min(daysAvailableE, key=lambda x:abs(x-int(dayEnd)))
        closestEndDate = f'{int(yearEnd)} {int(monthEnd)} {closestDayE}'
        print(closestEndDate)
        # Get the index of the start date in the data base
        for i, e in enumerate(df['startDate']):

            if e == str(closestStartDate):
                indexStart = i

        # Get the index of the end date in the data base
        for i, e in enumerate(df['endDate']):
            
            if e == str(closestEndDate):
                indexEnd = i
        
        # Crop the database with the obtained indexes
        df = df.iloc[indexStart:indexEnd]

elif timeFrame == 'd':
    # Having this option does not make sense now because we are working with subgroups and not discrete values

    span = input('All days (a), range of days (b) or specific day in all months (c): ')
    
    if span == 'b':
        startDate = str(input('Enter the starting date (YYYY-MM-DD hh:mm:ss): '))
        endDate = str(input('Enter the ending date (YYYY-MM-DD hh:mm:ss): '))
                
        df = df.loc[startDate:endDate]
    
    elif span == 'c':
        numberDays = int(input('Enter the day number: '))
        df = df.loc[df['day'] == numberDays]

# graphType = str(input('x bar (a), s chart (b), mR chart (c): '))
graphType = 'a'
if graphType == 'a':

    varGraph = 'mean'

elif graphType == 'b':

    varGraph = 'stds'

elif graphType == 'c':

    varGraph = 'ranges'

def splitter(df):
        
    if timeFrame == 'a':
        lenGroups = 372
    elif timeFrame == 'b':
        lenGroups = 31
    elif timeFrame == 'c':
        lenGroups = 7
    
    dates = []
    # New implementation
    if timeFrame == 'a':
            
        years = list(dict.fromkeys(df['year'].tolist()))
        
        dataList = []
        for i in years:
            
            df = df.loc[df['year'] == i]
            variable = df[varName].values.tolist()
            
            if len(variable) == 372:
                dataList.append(variable)
                dates.append(i)
            
            df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL']) 

        numberGroups = len(dataList)
        dataLength = math.floor(numberGroups*lenGroups)
        print('Data length: {} | Group lenght: {} | number of groups: {}'.format(dataLength, lenGroups, numberGroups))
    
    elif timeFrame == 'b':
        
        years = list(dict.fromkeys(df['year'].tolist()))
        months = list(dict.fromkeys(df['month'].tolist()))
        months.sort()
        
        if span == 'a' or span == 'b' or span == 'd' or span == 'e' or span == 'f':
        
            dataList = []
            for i in years:
            
                df = df.loc[df['year'] == i]
                
                for j in months:
                    
                    df = df.loc[df['month'] == j]
                    
                    if df.empty == True:
                        pass
                    elif df.empty == False:
                        
                        variable = df[varName].values.tolist()
                    
                        if len(variable) == 31:
                            dataList.append(variable)
                            dates.append(f'{j} {i}')
                    
                    df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL']) 

                    if j == 12 or j == monthEnd:
                        df = df.loc[df['year'] == (i+1)]
                    else:
                        df = df.loc[df['year'] == i]
                        
            numberGroups = len(dataList)
            dataLength = math.floor(numberGroups*lenGroups)
            print('Data length: {} | Group lenght: {} | number of groups: {}'.format(dataLength, lenGroups, numberGroups))

        elif span == 'c':
            
            dataList = []
            for i in years:
                
                df = df.loc[df['year'] == i]
                
                for j in months:
                    
                    df = df.loc[df['month'] == j]
                    
                    if df.empty == True:
                        pass
                    elif df.empty == False:
                        variable = df[varName].values.tolist()
                        
                        if len(variable) == 31:
                            dataList.append(variable)
                            dates.append(f'{j} {i}')
                        else:
                            pass
                    
                    df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])
            
            numberGroups = len(dataList)
            dataLength = math.floor(numberGroups*lenGroups)
            print('Data length: {} | Group lenght: {} | number of groups: {}'.format(dataLength, lenGroups, numberGroups))
    
    elif timeFrame == 'c':
        
        weeks = list(dict.fromkeys(df['week'].tolist()))
        weekOrder = list(dict.fromkeys(df['weekOrder'].tolist()))
        
        startDate = list(df['startDate'])
        endDate = list(df['endDate'])
        
        if span == 'a' or span == 'b':
            
            dataList = []
            for i in weeks:
                
                df = df.loc[df['week'] == i]
                
                if df.empty == True:
                    pass
                elif df.empty == False:
                    variable = df[varName].values.tolist()
                    
                    if len(variable) == 7:
                        dataList.append(variable)
                
                df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])

            # clean startDate and endDate
            startDate = [i for i in startDate if i != '-']
            endDate = [i for i in endDate if i != '-']
            
            for i in zip(startDate, endDate):
                dates.append(str(i))
            
            numberGroups = len(dataList)
            dataLength = math.floor(numberGroups*lenGroups)
            print('Data length: {} | Group lenght: {} | number of groups: {}'.format(dataLength, lenGroups, numberGroups))
        
        elif span == 'c':
            
            dataList = []
            
            def flatter(list):
                return [item for sublits in list for item in sublits]

            def toMatrix(l, n):
                return [l[i:i+n] for i in range(0, len(l), n)]
            
            for i in range(0, numberYears):

                # Start the variable
                daysAvailableS = []

                # Clean startDate and endDate
                startDate = [i for i in startDate if i != '-']
                # Get the closest starting date to the user input
                yearBegin = int(yearBeginOri) + i
                for i in startDate:
                    if int(i[0:4]) == yearBegin and int(i[5:7]) == int(monthBegin):
                        daysAvailableS.append(int(i[-2:]))
                            
            
                closestDayS = min(daysAvailableS, key=lambda x:abs(x-int(dayBegin)))
                closestStartDate = f'{int(yearBegin)} {int(monthBegin)} {closestDayS}'
                print(closestStartDate)
                
                for i, e in enumerate(df['startDate']):

                    if e == str(closestStartDate):
                        indexStart = i
                
                # The next implementation works if the data is contiuous, without any gaps. I would have to find the closest monday on the else: to fix it get the initial year and add i
                indexEnd = indexStart + 28*numberMonths # get the index of the end period
            
                # Crop the database with the obtained indexes
                df = df.iloc[indexStart:(indexEnd)]
                print(df)
                
                if df.empty == True:
                    pass
                elif df.empty == False:
                    variable = df[varName].values.tolist()
                    startDateC = df['startDate'].values.tolist()
                    endDateC = df['endDate'].values.tolist()
                    
                    # Group variable in nested lists of 7 items
                    data = toMatrix(variable, 7)
                    dataList.append(data)
                    
                startDateC = [i for i in startDateC if i != '-']
                endDateC = [i for i in endDateC if i != '-']
                
                for i in zip(startDateC, endDateC):
                    dates.append(str(i))
                
                df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])
            
            
            dataList = flatter(dataList)
            
            numberGroups = len(dataList)
            dataLength = math.floor(numberGroups*lenGroups)
            print('Data length: {} | Group lenght: {} | number of groups: {}'.format(dataLength, lenGroups, numberGroups))
            
        elif span == 'd':
            
            dataList = df[varName].values.tolist()
            dataList = [dataList[i:i + lenGroups] for i in range(0, len(dataList), lenGroups)]
            
            # clean startDate and endDate
            startDate = [i for i in startDate if i != '-']
            endDate = [i for i in endDate if i != '-']
            
            for i in zip(startDate, endDate):
                dates.append(str(i))
            
            numberGroups = len(dataList)
            dataLength = math.floor(numberGroups*lenGroups)
            print('Data length: {} | Group lenght: {} | number of groups: {}'.format(dataLength, lenGroups, numberGroups))
            
    elif timeFrame == 'd':
        
        # Having this option does not make sense now because we are working with subgroups and not discrete values
        dataLength = len(df)
        dataList = list(df[varName])
        mean = np.mean(dataList)
        
        if dataLength >= 20:
            lenGroups = math.floor(dataLength * 0.05)
            print(lenGroups)
            numberGroups = math.floor(dataLength/lenGroups)
            dataOut = dataLength - lenGroups*numberGroups
        
        elif dataLength < 20:
            lenGroups = 1
            numberGroups = dataLength
            dataOut = dataLength - lenGroups*numberGroups
        
        dataList = randomDeletter(dataList, dataOut, mean)
        # Takes a list and outputs a 2D list divided in subgroups of equal length
        dataList = [dataList[i:i + lenGroups] for i in range(0, len(dataList), lenGroups)]
        # dataList = np.array(dataList)
    
        numberGroups = len(dataList)
        dataLength = math.floor(numberGroups*lenGroups)
        print('Data length: {} | Group lenght: {} | number of groups: {}'.format(dataLength, lenGroups, numberGroups))

    return lenGroups, numberGroups, dataList, dates

def randomDeletter(inputList, n, mean):
    
    indextoDelete = set(random.sample(range(len(inputList)), n))

    valuestoDelete = []
    for i in indextoDelete:
        valuestoDelete.append(inputList[i])
    
    while valuestoDelete:
        for i in valuestoDelete:
            if (i < 0.90*mean) or (i > 1.10*mean): 
                to_delete = set(random.sample(range(len(inputList)), n))
                
                valuestoDelete = []
                for i in to_delete:
                    valuestoDelete.append(i)
            else:
                break
        break

    return [x for i,x in enumerate(inputList) if not i in indextoDelete]

class nelson:
    
    # first nelson rule
    def rule1(self, data, mean, std):
        sigmaUp = mean + 3*std
        sigmaDown = mean -3*std

        def isBetween(value, lower, upper):
            isBetween = value < upper and value > lower
            return 0 if isBetween else 1

        data['rule_1'] = dfchart.apply(lambda row: isBetween(row[varGraph], sigmaDown, sigmaUp), axis=1)

    # second nelson rule
    def rule2(self, data, mean):
        values = [0]*len(dfchart)
        
        # +1 means upside, -1 means downside
        upsideOrDownside = 0
        counter = 0
        for i in range(len(dfchart)):
            
            number = dfchart.iloc[i][varGraph]
            
            if number > mean:
                if upsideOrDownside == 1:
                    counter += 1
                else:
                    upsideOrDownside = 1
                    counter = 1
            elif number < mean:
                if upsideOrDownside == -1:
                    counter += 1
                else:
                    upsideOrDownside = -1
                    counter = 1
            
            if counter >= 9:
                values[i] = 1
        
        data['rule_2'] = values
    
    # third nelson rule
    def rule3(self, data):
        
        values = [0]*len(dfchart)
        
        previousNumber = dfchart.iloc[0][varGraph]
        # +1 means increasing, -1 means decreasing
        increasingOrDecreasing = 0
        counter = 0
        for i in range(1, len(dfchart)):
            number = dfchart.iloc[i][varGraph]
            if number > previousNumber:
                if increasingOrDecreasing == 1:
                    counter += 1
                else:
                    increasingOrDecreasing = 1
                    counter = 1
            elif number < previousNumber:
                if increasingOrDecreasing == -1:
                    counter += 1
                else:
                    increasingOrDecreasing = -1
                    counter = 1
            
            if counter >= 6:
                values[i] = 1
                
            previousNumber = number
                
        data['rule_3'] = values
    
    # fourth nelson rule
    def rule4(self, data):
        values = [0]*len(dfchart)
        
        previousNumber = dfchart.iloc[0][varGraph]
        # +1 means increasing, -1 means decreasing
        bimodal = 0
        counter = 1
        for i in range(1, len(dfchart)):
            number = dfchart.iloc[i][varGraph]
            
            if number > previousNumber:
                bimodal += 1
                if abs(bimodal) != 1:
                    counter = 0
                    bimodal = 0
                else:
                    counter +=1
            elif number < previousNumber:
                bimodal -= 1
                if abs(bimodal) != 1:
                    counter = 0
                    bimodal = 0
                else:
                    counter += 1
            
            previousNumber = number
            
            if counter >= 14:
                values[i] = 1
        
        data['rule_4'] = values
    
    # fifth nelson rule
    def rule5(self, data, mean, std):
        if len(dfchart) < 3: return
        
        values = [0]*len(dfchart)
        twosigmaUp = mean + 2*std
        twosigmaDown = mean - 2*std
        
        for i in range(len(dfchart) - 3):
            first = dfchart.iloc[i][varGraph]
            second = dfchart.iloc[i+1][varGraph]
            third = dfchart.iloc[i+2][varGraph]
            
            setValue = False
            validCounter = 0
            if first > mean and second > mean and third > mean:
                validCounter += 1 if first > twosigmaUp else 0
                validCounter += 1 if second > twosigmaUp else 0
                validCounter += 1 if third > twosigmaUp else 0
                setValue = validCounter >= 2
            elif first < mean and second < mean and third < mean:
                validCounter += 1 if first < twosigmaDown else 0
                validCounter += 1 if second < twosigmaDown else 0
                validCounter += 1 if third < twosigmaDown else 0
                setValue = validCounter >= 2
            
            if setValue:
                values[i+2] = 1
        
        data['rule_5'] = values
    
    # sixth nelson rule 
    def rule6(self, data, mean, std):
        if len(dfchart) < 5: return
        
        values = [0]*len(dfchart)
        onesigmaUp = mean - std
        onesigmaDown = mean + std
        
        for i in range(len(dfchart) - 5):
            pVals = list(map(lambda x: dfchart.iloc[x][varGraph], range(i, i+5)))
            
            setValue = False # revisar estas condiciones 
            if len(list(filter(lambda x: x > mean, pVals))) == 5:
                setValue = len(list(filter(lambda x: x > onesigmaDown, pVals))) >= 4
            elif len(list(filter(lambda x: x < mean, pVals))) == 5:
                setValue = len(list(filter(lambda x: x < onesigmaUp, pVals))) >= 4
                
            if setValue:
                values[i+4] = 1
                
        data['rule_6'] = values
    
    # fifth nelson rule
    def rule7(self, data, mean, std):
        if len(dfchart) < 15: return
        
        values = [0]*len(dfchart)
        onesigmaUp = mean + std
        onesigmaDown = mean - std
        
        for i in range(len(dfchart) - 15):
            setValue = True
            for y in range(15):
                item = dfchart.iloc[i + y][varGraph]
                if item >= onesigmaUp or item <= onesigmaDown:
                    setValue = False
                    break
            
            if setValue:
                values[i+14] = 1
        
        data['rule_7'] = values
    
    # eigth nelson rule
    def rule8(self, data, mean, std):
        if len(dfchart) < 8: return
        
        values = [0]*len(dfchart)
        
        for i in range(len(dfchart) - 8):
            setValue = True
            for y in range(8):
                item = dfchart.iloc[i + y][varGraph]
                if abs(mean - item) < std:
                    setValue = False
                    break
            
            if setValue:
                values[i+8] = 1
        
        data['rule_8'] = values

def format_arr(rule):
    
    rule_arr = 'rule_' + str(rule)
    return [index for index,val in enumerate(result[rule_arr]) if val]

def plotAxlines(array):
    
    mean = np.mean(array)
    std = np.std(array)
    colors = ['black','green','yellow','red']
    
    for level,color in enumerate(colors):
        upper = mean + std*level
        lower = mean - std*level
        plt.axhline(y=upper, linewidth=1, color=color)
        plt.axhline(y=lower, linewidth=1, color=color)
    
    return

def validator(rule, dates):
    
    outliers = []
    
    for i,j in zip(dates, result[str(rule)]):
        
        if j == 1:
            
            outliers.append(i)

    return outliers

if __name__ == '__main__':
    
    # adapt the dataset to processed
    lenGroups, numberGroups, dataList, dates = splitter(df)
    
    # define list variable for group means
    x = []

    # define list variable for group standard deviations
    s = []


    for group in dataList:
        x.append((np.mean(group)).round(3))
        s.append((np.std(group)).round(3))
    
    mrs = [[] for i in range(numberGroups)]

    # get and append moving ranges
    counter = 1
    for i,w in zip(dataList, range(len(mrs))):
        for j in range(len(i)):
            result = abs(round(i[j] - i[j-1], 2))
            mrs[w].append(result)
            counter += 1
            
            if counter == lenGroups:
                break
            
    # print(moving_ranges)
    
    # define list variable for the moving ranges
    mr = []
    for i in mrs:
        mr.append(round(np.mean(i), 4))

    # the historical mean for the process variable in question
    mean_x = (np.mean(x)).round(3)
    mean_s = (np.mean(s)).round(3)
    mean_mr = (np.mean(mr)).round(3)

    # the historical standard deviation for the process variable in question
    std_x = (np.std(x)).round(3)
    std_s = (np.std(s)).round(3)
    std_mr = (np.std(mr)).round(3)

    # create database with the key information for each graph
    dfchart = pd.DataFrame(list(zip(x, s, mr)), columns=['mean', 'stds', 'ranges'])
    
    
    if varGraph == 'mean':
        
        result = {'all_values': x,
                'rule_1': [],
                'rule_2': [],
                'rule_3': [],
                'rule_4': [],
                'rule_5': [],
                'rule_6': [],
                'rule_7': [],
                'rule_8': []
                }

        applier = nelson()

        applier.rule1(result, mean_x, std_x)
        applier.rule2(result, mean_x)
        applier.rule3(result)
        applier.rule4(result)
        applier.rule5(result, mean_x, std_x)
        applier.rule6(result, mean_x, std_x)
        applier.rule7(result, mean_x, std_x)
        applier.rule8(result, mean_x, std_x)

        mark = 6.0
        mean3top = mean_x + 3*std_x
        mean3low = mean_x - 3*std_x
        mean2top = mean_x + 2*std_x
        mean2low = mean_x - 2*std_x
        mean1top = mean_x + std_x
        mean1low = mean_x - std_x
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.set_title(r'$PM10$ '+r'$\bar{x}$'+f' chart')
        ax.set(xlabel='Date', ylabel=r'$\bar{x}' + r' (\mu*g/m^3)$')
        
        plt.plot(dates, result['all_values'], color='red', markevery=format_arr(1), ls='', marker='s', mfc='none', mec='red', label='Rule1', markersize=mark*5.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(2), ls='', marker='o', mfc='none', mec='blue', label="Rule2", markersize=mark*4.7)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(3), ls='', marker='o', mfc='none', mec='purple', label="Rule3", markersize=mark*4.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(4), ls='', marker='o', mfc='none', mec='green', label="Rule4", markersize=mark*3.7)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(5), ls='', marker='o', mfc='none', mec='orange', label='Rule5', markersize=mark*3.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(6), ls='', marker='o', mfc='none', mec='pink', label='Rule6', markersize=mark*2.7)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(7), ls='', marker='o', mfc='none', mec='cyan', label='Rule7', markersize=mark*2.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(8), ls='', marker='o', mfc='none', mec='olive', label='Rule8', markersize=mark*1.7)
        plt.plot(dates, result['all_values'], color='blue', marker="o", markersize=mark)
        plt.xticks(dates, rotation = 70)
        plt.grid(axis = 'x')

        left, right = ax.get_xlim()
        
        ax.text(right + 0.3, mean3top, r'$3\sigma$' + '=' + str('{:.2f}'.format(mean3top)), color='red')
        ax.text(right + 0.3, mean3low, r'$3\sigma$' + '=' + str('{:.2f}'.format(mean3low)), color='red')
        ax.text(right + 0.3, mean2top, r'$2\sigma$' + '=' + str('{:.2f}'.format(mean2top)), color='orange')
        ax.text(right + 0.3, mean2low, r'$2\sigma$' + '=' + str('{:.2f}'.format(mean2low)), color='orange')
        ax.text(right + 0.3, mean1top, r'$\sigma$' + '=' + str('{:.2f}'.format(mean1top)), color='green')
        ax.text(right + 0.3, mean1low, r'$\sigma$' + '=' + str('{:.2f}'.format(mean1low)), color='green')
        ax.text(right + 0.3, mean_x, r'$\bar{x}$' + '=' + str('{:.2f}'.format(mean_x)), color='black')
        
        plotAxlines(result['all_values'])

        if timeFrame == 'a' or timeFrame == 'b':
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(4))
            ax.xaxis.set_minor_locator(MultipleLocator(1))

        plt.legend()
        # plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.)


        # plt.savefig('control-chart.png')
        plt.show()
        
        # Validation
        rule = str(input('Which Weco rule do you wish to validate (rule_#): '))
        outliers = validator(rule, dates)
        
        print('outliers: ', outliers)
    
    if varGraph == 'stds':
        
        result = {'all_values': s,
                'rule_1': [],
                'rule_2': [],
                'rule_3': [],
                'rule_4': [],
                'rule_5': [],
                'rule_6': [],
                'rule_7': [],
                'rule_8': []
                }

        applier = nelson()

        applier.rule1(result, mean_s, std_s)
        applier.rule2(result, mean_s)
        applier.rule3(result)
        applier.rule4(result)
        applier.rule5(result, mean_s, std_s)
        applier.rule6(result, mean_s, std_s)
        applier.rule7(result, mean_s, std_s)
        applier.rule8(result, mean_s, std_s)
        
        mark = 6.0
        mean3top = mean_s + 3*std_s
        mean3low = mean_s - 3*std_s
        mean2top = mean_s + 2*std_s
        mean2low = mean_s - 2*std_s
        mean1top = mean_s + std_s
        mean1low = mean_s - std_s

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.set_title(r'$NO_2$' + ' s chart')
        ax.set(xlabel='Date', ylabel=r'$\sigma' + r' (\mu*g/m^3)$')
        
        plt.plot(dates, result['all_values'], color='red', markevery=format_arr(1), ls='', marker='s', mfc='none', mec='red', label='Rule1', markersize=mark*5.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(2), ls='', marker='o', mfc='none', mec='blue', label="Rule2", markersize=mark*4.7)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(3), ls='', marker='o', mfc='none', mec='purple', label="Rule3", markersize=mark*4.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(4), ls='', marker='o', mfc='none', mec='green', label="Rule4", markersize=mark*3.7)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(5), ls='', marker='o', mfc='none', mec='orange', label='Rule5', markersize=mark*3.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(6), ls='', marker='o', mfc='none', mec='pink', label='Rule6', markersize=mark*2.7)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(7), ls='', marker='o', mfc='none', mec='cyan', label='Rule7', markersize=mark*2.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(8), ls='', marker='o', mfc='none', mec='olive', label='Rule8', markersize=mark*1.7)
        plt.plot(dates, result['all_values'], color='blue', marker="o", markersize=mark)
        plt.xticks(dates, rotation= 70)
        plt.grid(axis = 'x')

        left, right = ax.get_xlim()
        
        ax.text(right + 0.3, mean3top, r'$3\sigma$' + '=' + str('{:.2f}'.format(mean3top)), color='red')
        ax.text(right + 0.3, mean3low, r'$3\sigma$' + '=' + str('{:.2f}'.format(mean3low)), color='red')
        ax.text(right + 0.3, mean2top, r'$2\sigma$' + '=' + str('{:.2f}'.format(mean2top)), color='orange')
        ax.text(right + 0.3, mean2low, r'$2\sigma$' + '=' + str('{:.2f}'.format(mean2low)), color='orange')
        ax.text(right + 0.3, mean1top, r'$\sigma$' + '=' + str('{:.2f}'.format(mean1top)), color='green')
        ax.text(right + 0.3, mean1low, r'$\sigma$' + '=' + str('{:.2f}'.format(mean1low)), color='green')
        ax.text(right + 0.3, mean_s, r'$\bar{x}$' + '=' + str('{:.2f}'.format(mean_s)), color='black')
        
        plotAxlines(result['all_values'])
    
        if timeFrame == 'a' or timeFrame == 'b':
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(4))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
        
        plt.legend()
        # plt.savefig('control-chart.png')
        plt.show()
        
        # Validation
        rule = str(input('Which Weco rule do you wish to validate (rule_#): '))
        outliers = validator(rule, dates)
        
        print('outliers: ', outliers)
    
    if varGraph == 'ranges':
        
        result = {'all_values': mr,
                'rule_1': [],
                'rule_2': [],
                'rule_3': [],
                'rule_4': [],
                'rule_5': [],
                'rule_6': [],
                'rule_7': [],
                'rule_8': []
                }

        applier = nelson()

        applier.rule1(result, mean_mr, std_mr)
        applier.rule2(result, mean_mr)
        applier.rule3(result)
        applier.rule4(result)
        applier.rule5(result, mean_mr, std_mr)
        applier.rule6(result, mean_mr, std_mr)
        applier.rule7(result, mean_mr, std_mr)
        applier.rule8(result, mean_mr, std_mr)
        
        mark = 6.0
        mean3top = mean_mr + 3*std_mr
        mean3low = mean_mr - 3*std_mr
        mean2top = mean_mr + 2*std_mr
        mean2low = mean_mr - 2*std_mr
        mean1top = mean_mr + std_mr
        mean1low = mean_mr - std_mr

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.set_title(r'$NO_2$' ' mR chart')
        ax.set(xlabel='Date', ylabel='Variability' + r'$(\mu*g/m^3)$')
        
        plt.plot(dates, result['all_values'], color='red', markevery=format_arr(1), ls='', marker='s', mfc='none', mec='red', label='Rule1', markersize=mark*5.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(2), ls='', marker='o', mfc='none', mec='blue', label="Rule2", markersize=mark*4.7)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(3), ls='', marker='o', mfc='none', mec='purple', label="Rule3", markersize=mark*4.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(4), ls='', marker='o', mfc='none', mec='green', label="Rule4", markersize=mark*3.7)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(5), ls='', marker='o', mfc='none', mec='orange', label='Rule5', markersize=mark*3.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(6), ls='', marker='o', mfc='none', mec='pink', label='Rule6', markersize=mark*2.7)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(7), ls='', marker='o', mfc='none', mec='cyan', label='Rule7', markersize=mark*2.2)
        plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(8), ls='', marker='o', mfc='none', mec='olive', label='Rule8', markersize=mark*1.7)
        plt.plot(dates, result['all_values'], color='blue', marker="o", markersize=mark)
        plt.xticks(dates, rotation = 70)
        plt.grid(axis = 'x')

        left, right = ax.get_xlim()
        
        ax.text(right + 0.3, mean3top, r'$3\sigma$' + '=' + str('{:.2f}'.format(mean3top)), color='red')
        ax.text(right + 0.3, mean3low, r'$3\sigma$' + '=' + str('{:.2f}'.format(mean3low)), color='red')
        ax.text(right + 0.3, mean2top, r'$2\sigma$' + '=' + str('{:.2f}'.format(mean2top)), color='orange')
        ax.text(right + 0.3, mean2low, r'$2\sigma$' + '=' + str('{:.2f}'.format(mean2low)), color='orange')
        ax.text(right + 0.3, mean1top, r'$\sigma$' + '=' + str('{:.2f}'.format(mean1top)), color='green')
        ax.text(right + 0.3, mean1low, r'$\sigma$' + '=' + str('{:.2f}'.format(mean1low)), color='green')
        ax.text(right + 0.3, mean_mr, r'$\bar{x}$' + '=' + str('{:.2f}'.format(mean_mr)), color='black')
        
        plotAxlines(result['all_values'])

        if timeFrame == 'a' or timeFrame == 'b':
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(4))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
    
        plt.legend()
        # plt.savefig('control-chart.png')
        plt.show()
        
        # Validation
        rule = str(input('Which Weco rule do you wish to validate (rule_#): '))
        outliers = validator(rule, dates)
        
        print('outliers: ', outliers)