
import os
import pandas as pd

"""This function acts on the processed database. Its main function
is to select the desired time frames by the user.

Returns:
    - dataMatrix: 2D list with the discrete data ready to be coverted to 
    functional data.
    - timeStamps: the time marks for each time frame."""

def toMatrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def builder(File, timeFrame):
    
    fileName, fileExtension = os.path.splitext(File)
    df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';') # Set column date as the index?
    cols = list(df.columns.values.tolist())

    # Select the data in the specified time frame
    years = list(dict.fromkeys(df['year'].tolist()))
    months = list(dict.fromkeys(df['month'].tolist()))
    months.sort()

    years = list(dict.fromkeys(df['year'].tolist()))
    months = list(dict.fromkeys(df['month'].tolist()))
    months.sort()

    weeks = list(dict.fromkeys(df['week'].tolist()))
    weeks = [i for i in weeks if i != 0]
    weekOrder = list(dict.fromkeys(df['weekOrder'].tolist()))

    startDate = list(df['startDate'])
    endDate = list(df['endDate'])

    days = list(dict.fromkeys(df['day'].tolist()))
    days.sort()

    if timeFrame == 'a':

        # Get information on the time frame desired by the user
        span = input('All months (a), all months in one or several given years (b), a specific month in every year (c), several months in every year (d) several months in several years (e) or a range of months (f): ')

        if span == 'b':
            numberYears = list(map(int, input('Enter the years (space-separated): ').split()))
        
        elif span == 'c':
            numberMonths = list(map(int, input('Enter the month number: ')))
        
        elif span == 'd':
            numberMonths = list(map(int, input('Enter the months (space-separated): ').split()))
        
        elif span == 'e':
            numberYears = list(map(int, input('Enter the years (space-separated): ').split()))
            numberMonths = list(map(int, input('Enter the months (space-separated): ').split()))
        
        elif span == 'f':
            monthStart = int(input('Enter the starting month: '))
            yearStart = int(input('Enter the starting year: '))
            monthEnd = int(input('Enter the ending month: '))
            yearEnd = int(input('Enter the ending year: '))

        # Initialize two lists for dataMatrix and timeStamps
        dataMatrix = []
        timeStamps =  []

        if span == 'a': # All months available in the database

            # Put the desired data in dataMatrix  and timeStamps
            for i in years:

                df = df.loc[df['year'] == i]

                for j in months:

                    df = df.loc[df['month'] == j]

                    if df.empty == True:
                        pass
                    elif df.empty == False:
                        variable = df['value'].values.tolist()

                        if len(variable) == 2976:
                            dataMatrix.append(variable)
                            timeStamps.append(f'{j} {i}')
                        else:
                            pass

                    df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')

                    if j == 12:
                        df = df.loc[df['year'] == (i+1)]
                    else:
                        df = df.loc[df['year'] == i]

        elif span == 'b': # All months in one or several given years

            # Put the desired data in dataMatrix and timeStamps
            for i in numberYears:

                df = df.loc[df['year'] == i]

                for j in months:

                    df = df.loc[df['month'] == j]

                    if df.empty == True:
                        pass
                    elif df.empty == False:
                        variable = df['value'].values.tolist()

                        if len(variable) == 2976:
                            dataMatrix.append(variable)
                            timeStamps.append(f'{j} {i}')
                        else:
                            pass

                    df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')

                    if j == 12:
                        df = df.loc[df['year'] == (i+1)]
                    else:
                        df = df.loc[df['year'] == i]
            
        elif span == 'c': # a specific month in every year

            for i in years:

                df = df.loc[df['year'] == i]

                for j in numberMonths:

                    df = df.loc[df['month'] == j]

                    if df.empty == True:
                        pass
                    elif df.empty == False:
                        variable = df['value'].values.tolist()

                        if len(variable) == 2976:
                            dataMatrix.append(variable)
                            timeStamps.append(f'{j} {i}')
                        else:
                            pass
                    
                    df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')

        elif span == 'd': # several months in every year

            for i in years:
                
                df = df.loc[df['year'] == i]

                for j in numberMonths:

                    df = df.loc[df['month'] == j]

                    if df.empty == True:
                        pass
                    elif df.empty == False:
                        variable = df['value'].values.tolist()

                        if len(variable) == 2976:
                            dataMatrix.append(variable)
                            timeStamps.append(f'{j} {i}')
                        else:
                            pass

                    df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')

                    if j == numberMonths[-1]:
                        df = df.loc[df['year'] == (i+1)]
                    else:
                        df = df.loc[df['year'] == i]

        elif span == 'e': # several months in several years

            for i in numberYears:

                df = df.loc[df['year'] == i]

                for j in numberMonths:

                    df = df.loc[df['month'] == j]
                    
                    if df.empty == True:
                        pass
                    elif df.empty == False:
                        variable = df['value'].values.tolist()

                        if len(variable) == 2976:
                            dataMatrix.append(variable)
                            timeStamps.append(f'{j} {i}')
                        else:
                            pass
                    
                    df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
                    
                    if j != numberMonths[-1]:
                        df = df.loc[df['year'] == i]

        elif span == 'f': # a range of months
            
            for i in years:
                
                if i >= yearStart:
                    
                    df = df.loc[df['year'] == i]
                    
                    for j in months:
                        
                        if j >= monthStart and i <= yearEnd:
                            
                            df = df.loc[df['month'] == j]
                            
                            if df.empty == True:
                                pass
                            elif df.empty == False:
                                variable = df['value'].values.tolist()
                                
                                if len(variable) == 2976:
                                    dataMatrix.append(variable)
                                    timeStamps.append(f'{j} {i}')
                                else:
                                    pass
                                    
                            df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')

                            if j == 12:
                                df = df.loc[df['year'] == (i+1)]
                            else:
                                df = df.loc[df['year'] == i]
                            
                            if j == monthEnd and i == yearEnd:
                                break


    elif timeFrame == 'b':

        # Get information on the time frame desired by the user
        span = input('All weeks (a), 1st/2nd/3rd/4th week of each month (b), range of weeks in several or all years (c) or range of weeks (d): ')

        if span == 'b':
            weekNumber = list(map(int, input('Enter the week number: ')))
        
        elif span == 'c':
            yearBeginOri, monthBegin, dayBegin = input('Enter the first year desired: '), input('Enter the first month desired: '), input('Enter the first day desired: ')
            yearEndOri, monthEnd, dayEnd = input('Enter the last year desired: '), input('Enter the last month desired: '), input('Enter the last day desired: ')
        
        elif span == 'd':
            yearBegin, monthBegin, dayBegin = input('Enter the first year desired: '), input('Enter the first month desired: '), input('Enter the first day desired: ')
            yearEnd, monthEnd, dayEnd = input('Enter the last year desired: '), input('Enter the last month desired: '), input('Enter the last day desired: ')

        # Initialize two lists for dataMatrix and timeStamps
        dataMatrix = []
        timeStamps =  []
        
        if span == 'a': # all weeks available in the data base
            
            for i in weeks:
                
                df = df.loc[df['week'] == i]
                
                if df.empty == True:
                    pass
                elif df.empty == False:
                    variable = df['value'].values.tolist()
                    
                    if len(variable) == 672:
                        dataMatrix.append(variable)
                    else:
                        pass
                
                df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
                
            # Clean startDate and endDate
            startDate = [i for i in startDate if i != '-']
            endDate = [i for i in endDate if i != '-']
            
            for i in zip(startDate, endDate):
                timeStamps.append(str(i))
            
        elif span == 'b': # 1st/2nd/3rd/4th week of each month
            
            for i in weekNumber:
                
                df = df.loc[df['week'] == i]
                
                if df.empty == True:
                    pass
                elif df.empty == False:
                    variable = df['value'].values.tolist()
                    startDateB = df['startDate'].values.tolist()
                    endDateB = df['endDate'].values.tolist()
                    
                    # group variable in nested lists of 672 items. Look into this
                    dataMatrix = toMatrix(variable, 672) # no flatter() needed in this case
                    dataMatrix = [i for i in dataMatrix if len(i) == 672]
                
                df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
                
            # clean startDate and endDate
            startDateB = [i for i in startDateB if i != '-']
            endDateB = [i for i in endDateB if i != '-']
            
            for i in zip(startDateB, endDateB):
                timeStamps.append(str(i))
        
        elif span == 'c': # range of weeks in several or all years
            
            numberYears = int(yearEndOri) - int(yearBeginOri)
            
            for i in range(numberYears):
                
                # Start the variable
                daysAvailableS = []
                daysAvailableE = []
                
                # Clean startDate and endDate
                startDate = [i for i in startDate if i != '-']
                endDate = [i for i in endDate if i != '-']
                
                # Get the closest starting date to the user input
                yearBegin = int(yearBeginOri) + i
                
                # RFE: In case the initial or ending months are missing it should do the following:
                # For the initial month -> check the starting dates in the following month and do the same until a existing startind date is found
                # For the ending month -> the same but backwards in time.
                for j in startDate:
                    # It should jump ot the next month before the if statement in case the initial month does not exist
                    if int(j[0:4]) == yearBegin and int(j[5:7]) == int(monthBegin):
                        daysAvailableS.append(int(j[-2:]))
                    
                closestDayS = min(daysAvailableS, key=lambda x:abs(x-int(dayBegin)))
                closestStartDate = f'{int(yearBegin)} {int(monthBegin)} {closestDayS}'
                print(closestStartDate)
                
                # Get the closest end date to the users input
                yearEnd = yearBegin
                
                for j in endDate:
                
                    if int(j[0:4]) == yearEnd and int(j[5:7])== int(monthEnd):
                        daysAvailableE.append(int(j[-2:]))
                
                closestDayE = min(daysAvailableE, key=lambda x:abs(x-int(dayEnd)))
                closestEndDate = f'{int(yearEnd)} {int(monthEnd)} {closestDayE}'
                print(closestEndDate)
                
                # Get the index of the start date in the database
                for i, e in enumerate(df['startDate']):

                    if e == str(closestStartDate):
                        indexStart = i

                # Get the index of the end date in the database
                for i, e in enumerate(df['endDate']):
                    if e == str(closestEndDate):
                        indexEnd = i
                
                # Crop the database with the obtained indexes
                df = df.iloc[indexStart:indexEnd]
                
                if df.empty == True:
                    pass
                elif df.empty == False:
                    variable = df['value'].values.tolist()
                    startDateC = df['startDate'].values.tolist()
                    endDateC = df['endDate'].values.tolist()
                
                    dataMatrix = [[] for i in range(len(variable))] # With toMatrix() this is not needed
                    
                    # group variable in nested lists of 672 items
                    dataMatrix = toMatrix(variable, 672)
                
                startDateC = [i for i in startDateC if i != '-']
                endDateC = [i for i in endDateC if i != '-']
                
                for i in zip(startDateC, endDateC):
                    timeStamps.append(str(i))
        
        elif span == 'd': # range of weeks
            
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
            # Get the closest end date to the users input
            for i in endDate:
                
                if i[0:4] == yearEnd and int(i[5:7])== int(monthEnd):
                    daysAvailableE.append(int(i[-2:]))
            
            closestDayE = min(daysAvailableE, key=lambda x:abs(x-int(dayEnd)))
            closestEndDate = f'{int(yearEnd)} {int(monthEnd)} {closestDayE}'
            print(closestEndDate)
            # Get the index of the start date in the database
            for i, e in enumerate(df['startDate']):

                if e == str(closestStartDate):
                    indexStart = i

            # Get the index of the end date in the database
            for i, e in enumerate(df['endDate']):
                if e == str(closestEndDate):
                    indexEnd = i
            
            # Crop the database with the obtained indexes
            df = df.iloc[indexStart:indexEnd]
            
            if df.empty == True:
                pass
            elif df.empty == False:
                variable = df['value'].values.tolist()
                startDateC = df['startDate'].values.tolist()
                endDateC = df['endDate'].values.tolist()

                # group variable in nested lists of 672 items
                dataMatrix = toMatrix(variable, 672)

            startDateC = [i for i in startDateC if i != '-']
            endDateC = [i for i in endDateC if i != '-']
            
            for i in zip(startDateC, endDateC):
                timeStamps.append(str(i))


    elif timeFrame == 'c':
        
        span = input('All days (a), all days of one or several years (b) all days of one or several months (c), one or a range of days in every month of every year (d), one or a range of days in several years (e), one or a range of days in several months in several years (f), and a range of days (g): ')
        
        if span == 'b':
            numberYears = list(map(int, input('Enter the years (space-separated): ').split()))
        
        elif span == 'c':
            numberMonths = list(map(int, input('Enter the months (space-separated): ').split()))
        
        elif span == 'd':
            numberDays = list(map(int, input('Enter the days (space-separated): ').split()))
        
        elif span == 'e':
            numberYears = list(map(int, input('Enter the years (space-separated): ').split()))
            numberDays = list(map(int, input('Enter the days (space-separated): ').split()))
        
        elif span == 'f':
            numberYears = list(map(int, input('Enter the years (space-separated): ').split()))
            numberMonths = list(map(int, input('Enter the months (space-separated): ').split()))
            numberDays = list(map(int, input('Enter the days (space-separated): ').split()))
        
        elif span == 'g':
            startDate = str(input('Enter the starting date (YYYY-MM-DD hh:mm:ss): '))
            endDate = str(input('Enter the ending date (YYYY-MM-DD hh:mm:ss): '))
        
        # Initialize two lists for dataMatrix and timeStamps
        dataMatrix = []
        timeStamps =  []
        
        if span == 'a': # all days
            
            for i in years:
                
                df = df.loc[df['year'] == i]
                
                for j in months:
                    
                    df = df.loc[df['month'] == j]

                    for k in days:
                        
                        df = df.loc[df['day'] == k]
                        
                        if df.empty == True:
                            pass
                        elif df.empty == False:
                            variable = df['value'].values.tolist()
                            
                            if len(variable) == 96:
                                dataMatrix.append(variable)
                                timeStamps.append(f'{k} {j} {i}')
                            else:
                                pass

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
        
        if span == 'b': # all days of one or several years
                
            for e, i in enumerate(numberYears):
                
                df = df.loc[df['year'] == i]
                
                for j in months:
                    
                    df = df.loc[df['month'] == j]
                    
                    for k in days:
                    
                        df = df.loc[df['day'] == k]
                        
                        if df.empty == True:
                            pass
                        elif df.empty == False:
                            variable = df['value'].values.tolist()
                            
                            if len(variable) == 96:
                                dataMatrix.append(variable)
                                timeStamps.append(f'{k} {j} {i}')
                            else:
                                pass
                    
                        df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
                        
                        if j == 12 and k == 31 and (i != numberYears[-1]):
                            df = df.loc[df['year'] == (numberYears[e+1])]
                            df = df.loc[df['month'] == 1]
                        
                        elif j <= 12:
                            df = df.loc[df['year'] == i]
                            
                            if k < 31:
                                df = df.loc[df['month'] == j]
                            
                            elif k == 31:
                                df = df.loc[df['month'] == (j+1)]
        
        elif span == 'c': # all days of one or several months
            
            for i in years:
                
                df = df.loc[df['year'] == i]
                
                for j in numberMonths:
                
                    df = df.loc[df['month'] == j]
                    
                    for k in days:
                        
                        df = df.loc[df['day'] == k]
                        
                        if df.empty == True:
                            pass
                        elif df.empty == False:
                            variable = df['value'].values.tolist()
                            
                            if len(variable) == 96:
                                dataMatrix.append(variable)
                                timeStamps.append(f'{k} {j} {i}')
                                print(len(df), f'{k} {j} {i}')
                            else:
                                pass

                        df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
                        
                        if j == numberMonths[-1] and k == 31:
                            df = df.loc[df['year'] == (i+1)]
                            df = df.loc[df['month'] == numberMonths[0]]
                            
                        else:
                            df = df.loc[df['year'] == i]
                            
                            if k < 31:
                                df = df.loc[df['month'] == j]
                            
                            elif k == 31:
                                df = df.loc[df['month'] == (j+1)]
        
        elif span == 'd': # One or a range of days in every month of every year
            
            for i in years:
                
                df = df.loc[df['year'] == i]
                
                for j in months:
                
                    df = df.loc[df['month'] == j]
                    
                    for k in numberDays:
                        
                        df = df.loc[df['day'] == k]
                        
                        if df.empty == True:
                            pass
                        elif df.empty == False:
                            variable = df['value'].values.tolist()
                            
                            if len(variable) == 96:
                                dataMatrix.append(variable)
                                timeStamps.append(f'{k} {j} {i}')
                            else:
                                pass

                        df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
                        
                        if j == 12 and k == numberDays[-1]:
                            df = df.loc[df['year'] == (i+1)]
                            df = df.loc[df['month'] == 1]
                        
                        elif j <= 12:
                            df = df.loc[df['year'] == i]
                            
                            if k < numberDays[-1]:
                                df = df.loc[df['month'] == j]
                            
                            elif k == numberDays[-1]:
                                df = df.loc[df['month'] == (j+1)]
        
        elif span == 'e': # one or a range of days in several years
            
            for e, i in enumerate(numberYears):
                
                df = df.loc[df['year'] == i]
                
                for j in months:
                
                    df = df.loc[df['month'] == j]
                    
                    for k in numberDays:
                        
                        df = df.loc[df['day'] == k]
                        
                        if df.empty == True:
                            pass
                        elif df.empty == False:
                            variable = df['value'].values.tolist()
                            
                            if len(variable) == 96:
                                dataMatrix.append(variable)
                                timeStamps.append(f'{k} {j} {i}')
                            else:
                                pass

                        df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
                        
                        if j == 12 and k == numberDays[-1] and (i != numberYears[-1]):
                            df = df.loc[df['year'] == (numberYears[e+1])]
                            df = df.loc[df['month'] == 1]
                        
                        elif j <= 12:
                            df = df.loc[df['year'] == i]
                            
                            if k < numberDays[-1]:
                                df = df.loc[df['month'] == j]
                            
                            elif k == numberDays[-1]:
                                df = df.loc[df['month'] == (j+1)]
        
        elif span == 'f': # one or a range of days in several months in several years
            
            for e, i in enumerate(numberYears):
                
                df = df.loc[df['year'] == i]
                
                for f, j in enumerate(numberMonths):
                
                    df = df.loc[df['month'] == j]
                    
                    for k in numberDays:
                        
                        df = df.loc[df['day'] == k]
                        
                        if df.empty == True:
                            pass
                        elif df.empty == False:
                            variable = df['value'].values.tolist()
                            
                            if len(variable) == 96:
                                dataMatrix.append(variable)
                                timeStamps.append(f'{k} {j} {i}')
                            else:
                                pass

                        df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';')
                        
                        if j == numberMonths[-1] and k == numberDays[-1] and (i != numberYears[-1]):
                            df = df.loc[df['year'] == (numberYears[e+1])]
                            df = df.loc[df['month'] == numberMonths[0]]
                        
                        elif j <= numberMonths[-1]:
                            df = df.loc[df['year'] == i]
                            
                            if k < numberDays[-1]:
                                df = df.loc[df['month'] == j]
                            
                            elif k == numberDays[-1] and (j != numberMonths[-1]):
                                df = df.loc[df['month'] == (numberMonths[f+1])]
        
        elif span == 'g': # a range of days
            
            # Get the indices of the start and ending dates
            startDateIndex = int(df.index[df['date'] == startDate].values)
            endDateIndex = int(df.index[df['date'] == endDate].values)
            
            df = df.loc[startDateIndex:endDateIndex]
            
            # Create dataMatrix and timeStamps
            variable = df['value'].values.tolist()
            dataMatrix = toMatrix(variable, 96)
            
            for i, j, k in zip(df['year'], df['month'], df['day']):
                timeStamps.append(f'{k} {j} {i}')
            
            timeStamps = list(dict.fromkeys(timeStamps)) # Remove repeated items

    return dataMatrix, timeStamps

