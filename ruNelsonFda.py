import os
import math
import random
import numpy as np
import skfda as fda
import pandas as pd
import plotly.graph_objects as go
from  matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
from matplotlib.ticker import MultipleLocator

def flatter(list):
    return [item for sublits in list for item in sublits]

def toMatrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def dataGrid(dataMatrix, timeFrame):

    # Define object FDataGrid
    if timeFrame == 'a':
        gridPoints = list(range(372)) # solución temporal para sacar gridPoints
    elif timeFrame == 'b':
        gridPoints = list(range(31))
    elif timeFrame == 'c':
        gridPoints = list(range(7))

    
    functionalData = fda.FDataGrid(data_matrix=dataMatrix, grid_points=gridPoints)
    
    # fig, axes = plt.subplots()
    
    # functionalData.plot(axes=axes)
    
    # axes.set_title(f'Original data {varName}')
    # axes.set_xlabel('Days')
    # axes.set_ylabel(f'{varName}')
    # fig.savefig(f'original_{varName}.png')
    
    return functionalData, gridPoints, dataMatrix

def smoothing(functionalData, dataMatrix):

    # Calculate Fourier smoothing and the number of basis functions
    dataMatrixFlat = flatter(dataMatrix) # flatten dataMatrix so it can be used to get Pearson's coef.
    
    for nBasis in range(200):
        
        basis = fda.representation.basis.Fourier(n_basis=nBasis) # fitting the data through Fourier
        smoothedData = functionalData.to_basis(basis) # Change class to FDataBasis
        
        evaluatingPoints = smoothedData.evaluate(np.array(gridPoints), derivative=0) # get the corresponding values in the resulting curve
        evaluatingPoints = evaluatingPoints.tolist() # convert from array to list

        flat2evaluatingPoints = flatter(evaluatingPoints)
        flatevaluatingPoints = flatter(flat2evaluatingPoints)
    
        rho, p = pearsonr(np.array(dataMatrixFlat), np.array(flatevaluatingPoints))
        
        if rho >= 0.95 or nBasis >= len(dataMatrix[0]):
            break
        else:
            continue
    
    print('Number of basis functions: ', nBasis, 'and rho: ', rho)
    
    # fig, axes = plt.subplots()
    
    # smoothedData.plot(axes=axes)
    # axes.set_title(f'Smoothed data {varName}')
    # axes.set_xlabel('Days')
    # axes.set_ylabel(f'{varName}')
    # fig.savefig(f'smoothed_{varName}.png')
    
    return smoothedData

def depthCalc(smoothedData, gridPoints):
    
    # Depth calculation
    smoothedDataGrid = smoothedData.to_grid(grid_points=gridPoints) # Convert to FDataGrid

    depthMethodInt = fda.exploratory.depth.IntegratedDepth()
    depthInt = depthMethodInt(smoothedDataGrid)
    
    depthMethodMBD = fda.exploratory.depth.ModifiedBandDepth()
    depthMBD = depthMethodMBD(smoothedDataGrid)
    
    depthMethodProj = fda.exploratory.depth.multivariate.ProjectionDepth()
    # depthProj = depthMethodProj(smoothedDataGrid)
    
    depthMethodSimpli = fda.exploratory.depth.multivariate.SimplicialDepth()
    # depthSimpli = depthMethodSimpli(smoothedDataGrid)
    
    # print("Depth:", depth)
    
    return smoothedDataGrid, depthInt, depthMBD

def msplot(smoothedData, smoothedDataGrid, depth, cutoff):
    
    color, outliercolor = 0.3, 0.7
    
    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smoothedDataGrid, multivariate_depth=depth, cutoff_factor=cutoff)

    outliersMSPlot = funcMSPlot.outliers
    
    # Double filter
    if len(outliersMSPlot) != 0:
        
        # Take the two dimensions separate
        mag = funcMSPlot.points[:, 0]
        shape = funcMSPlot.points[:, 1]

        mag90 = np.where((mag < np.percentile(mag, 7)) | (mag > np.percentile(mag, 93)))
        shape90 = np.where(shape > np.percentile(shape, 85))
        
        # Implementacion elipse
        b = np.percentile(shape, 85)
        a = np.percentile(mag, 80) - np.percentile(mag, 20)
        elip = np.where(1 < (((pow((mag), 2) // pow(a, 2)) + (pow((shape), 2) // pow(b, 2)))))
        

        colors = np.copy(outliersMSPlot).astype(float)
        colors[:] = color
        # colors[mag90] = outliercolor
        # colors[shape90] = outliercolor
        colors[elip] = outliercolor

        labels = np.copy(funcMSPlot.outliers.astype(int))
        labels[:] = 0
        # labels[mag90] = 1
        # labels[shape90] = 1
        labels[elip] = 1
    
    else:
        labels = []
    
        
    outliers = list(np.copy(outliersMSPlot).astype(int))
    
    print('outliers:',  outliers)
    
    outliersBoosted = list(labels)
    
    print('outliers test boosted:',  outliersBoosted)
    
    
    return outliers, outliersBoosted

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

def format_arr_fda(data):
    
    return [index for index,val in enumerate(data) if not val]

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

file = 'argentina2_pro.csv'
fileName, fileExtension = os.path.splitext(f'DataBase\{file}')

# Load the data frame and get the columns
df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL']) 
cols = list(df.columns.values.tolist())
dates = list(df.index)
print('Variable names: ', cols[3:-4])

# Cutoff params
cutoffIntBox = 1
cutoffMDBBox = 1
cutoffIntMS = 0.993
cutoffMDBMS = 0.993

# Define depths here
integratedDepth = fda.exploratory.depth.IntegratedDepth().multivariate_depth
modifiedbandDepth = fda.exploratory.depth.ModifiedBandDepth().multivariate_depth
projectionDepth = fda.exploratory.depth.multivariate.ProjectionDepth()
simplicialDepth = fda.exploratory.depth.multivariate.SimplicialDepth()


# Start questionnaire to know what to plot:
# varName = str(input('Insert the name of the variable as it is in the database: '))
varName = 'no2'
# timeFrame = input('Years (a), months (b) or weeks (c): ')
timeFrame = 'c'

if timeFrame == 'a':
    
    span = input('Use a range of years (Y/n): ') 
    
    if span == 'Y':
        numberYears = list(map(int, input('Enter the years within the range desired (space-separated): ').split()))
    else:
        pass
    
    years = list(dict.fromkeys(df['year'].tolist()))

    if span == 'n':
        
        timeStamps = []
        for i in years:
            
            df = df.loc[df['year'] == i]
            variable = df[varName].values.tolist()
            
            if len(variable) == 372:
                timeStamps.append(i)
            
            df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL']) 
    
    if span == 'Y':
        years = [x for x in numberYears if x in years]
        timeStamps = years
    else:
        pass
    
    dataMatrix = []    
        
    # Get data for each year and put it into dataMatrix
    for i in years: 
        
        df = df.loc[df['year'] == i]
        variable = df[varName].values.tolist()
        
        if len(variable) == 372:
            dataMatrix.append(variable)
        else:
            pass
        
        df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL']) 

elif timeFrame == 'b':

    # span = input('All months (a), all months in one or several given years (b), a specific month in every year (c), several months in every year (d) several months in several years (e) or a range of months (f): ')
    span = 'a'
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
    else:
        pass
    
    years = list(dict.fromkeys(df['year'].tolist()))
    months = list(dict.fromkeys(df['month'].tolist()))
    months.sort()
    timeStamps =  []
    
    if span == 'a':
        
        # Create empty data matrix
        dataMatrix = []

        # Put the desired data in dataMatrix
        for i in years:
            
            df = df.loc[df['year'] == i]

            for j in months:
                
                df = df.loc[df['month'] == j]

                if df.empty == True:
                    pass
                elif df.empty == False:
                    variable = df[varName].values.tolist()
                    
                    if len(variable) == 31:
                        dataMatrix.append(variable)
                        timeStamps.append(f'{j} {i}')
                    else:
                        pass
                    
                df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])
                
                if j == 12:
                    df = df.loc[df['year'] == (i+1)]
                else:
                    df = df.loc[df['year'] == i]
    
    elif span == 'b':
        
        # Create empty data matrix
        dataMatrix = [] 

        # Put the desired data in dataMatrix
        for i in years:
            
            if i in numberYears:

                df = df.loc[df['year'] == i]

                for j in months:
                    
                    df = df.loc[df['month'] == j]

                    if df.empty == True:
                        pass
                    elif df.empty == False:
                        variable = df[varName].values.tolist()
                        
                        if len(variable) == 31:
                            dataMatrix.append(variable)
                            timeStamps.append(f'{j} {i}')
                        else:
                            pass
                        
                    df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])
                    
                    if j == 12:
                        df = df.loc[df['year'] == (i+1)]
                    else:
                        df = df.loc[df['year'] == i]
    
    elif span == 'c':

        # Create empty data matrix
        dataMatrix = [] 

        for i in years:
    
            df = df.loc[df['year'] == i]
    
            for j in numberMonths:
            
                df = df.loc[df['month'] == j]
                
                if df.empty == True:
                    pass
                elif df.empty == False:
                    variable = df[varName].values.tolist()
                    
                    if len(variable) == 31:
                        dataMatrix.append(variable)
                        timeStamps.append(f'{j} {i}')
                    else:
                        pass
                    
                df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])
        
    elif span == 'd':

        # Create empty data matrix
        dataMatrix = []

        # Put the desired data in dataMatrix
        for i in years:

            df = df.loc[df['year'] == i]

            for j in numberMonths:

                df = df.loc[df['month'] == j]
                
                if df.empty == True:
                    pass
                elif df.empty == False:
                    variable = df[varName].values.tolist()
                    
                    if len(variable) == 31:
                        dataMatrix.append(variable)
                        timeStamps.append(f'{j} {i}')
                    else:
                        pass
                    
                df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])

                if j == numberMonths[-1]:
                    df = df.loc[df['year'] == (i+1)]
                else:
                    df = df.loc[df['year'] == i]

    elif span == 'e':

        # Create empty data matrix
        dataMatrix = []

        # put the desired data in dataMatrix
        for i in numberYears:

            df = df.loc[df['year'] == i]

            for j in numberMonths:

                df = df.loc[df['month'] == j]
                
                if df.empty == True:
                    pass
                elif df.empty == False:
                    variable = df[varName].values.tolist()
                    
                    if len(variable) == 31:
                        dataMatrix.append(variable)
                        timeStamps.append(f'{j} {i}')
                    else:
                        pass
                    
                df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])

                if j != numberMonths[-1]:
                    df = df.loc[df['year'] == i]
    
    elif span == 'f':

        # Create empty data matrix
        dataMatrix = [] # number Months in stead of months 

        # Put the desired data in dataMatrix
        for i in years:
            
            if i >= yearStart:

                df = df.loc[df['year'] == i]

                for j in months:
                    
                    if j >= monthStart and i <= yearEnd:
                    
                        df = df.loc[df['month'] == j]

                        if df.empty == True:
                            pass
                        elif df.empty == False:
                            variable = df[varName].values.tolist()
                            
                            if len(variable) == 31:
                                dataMatrix.append(variable)
                                timeStamps.append(f'{j} {i}')
                            else:
                                pass
                            
                        df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])
                        
                        if j == 12:
                            df = df.loc[df['year'] == (i+1)]
                        else:
                            df = df.loc[df['year'] == i]
                    
                        if j == monthEnd and i == yearEnd:
                            break

elif timeFrame == 'c':
    
    # span = input('All weeks (a), 1st/2nd/3rd/4th week of each month (b), range of weeks in several or all years (c) or range of weeks (d): ')
    span = 'c'
    if span == 'b':
        weekNumber = list(map(int, input('Enter the week number: ')))
    elif span == 'c':
        # yearBeginOri, monthBegin, dayBegin = input('Enter the first year desired: '), input('Enter the first month desired: '), input('Enter the first day desired: ')
        # numberYears, numberMonths = int(input('Enter number of years to analyze: ')), int(input('Enter the number of months to analyze: '))
        yearBeginOri, monthBegin, dayBegin = '2014', '1', '1'
        numberYears, numberMonths = 8, 4
    elif span == 'd':
        yearBegin, monthBegin, dayBegin = input('Enter the first year desired: '), input('Enter the first month desired: '), input('Enter the first day desired: ')
        yearEnd, monthEnd, dayEnd = input('Enter the last year desired: '), input('Enter the last month desired: '), input('Enter the last day desired: ')
    
    weeks = list(dict.fromkeys(df['week'].tolist()))
    weekOrder = list(dict.fromkeys(df['weekOrder'].tolist()))
    
    startDate = list(df['startDate'])
    endDate = list(df['endDate'])
    
    timeStamps = []
    
    if span == 'a':
        
        # Create empty data matrix
        dataMatrix = []
        
        for i in weeks:
            
            df = df.loc[df['week'] == i]
            
            if df.empty == True:
                pass
            elif df.empty == False:
                variable = df[varName].values.tolist()
                
                if len(variable) == 7: # time stamps are missing here
                    dataMatrix.append(variable)
                else:
                    pass
            
            df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])

        # clean startDate and endDate
        startDate = [i for i in startDate if i != '-']
        endDate = [i for i in endDate if i != '-']
        
        for i in zip(startDate, endDate):
            timeStamps.append(str(i))
    
    elif span == 'b':
        
        for i in weekNumber:
            
            df = df.loc[df['weekOrder'] == i]
            
            if df.empty == True:
                pass
            elif df.empty == False:
                variable = df[varName].values.tolist()
                startDateB = df['startDate'].values.tolist()
                endDateB = df['endDate'].values.tolist()
                
                # group variable in nested lists of 7 items. Look into this
                dataMatrix = toMatrix(variable, 7) # no flatter() needed in this case
                dataMatrix = [i for i in dataMatrix if len(i) == 7] 
                
            df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])

        # clean startDate and endDate
        startDateB = [i for i in startDateB if i != '-']
        endDateB = [i for i in endDateB if i != '-']
        
        for i in zip(startDateB, endDateB):
            timeStamps.append(str(i))
    
    elif span == 'c':
        
        dataMatrix = []

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
            # nextIndex = indexStart + 372 # leap a whole year
        
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
                dataMatrix.append(data)
                
            startDateC = [i for i in startDateC if i != '-']
            endDateC = [i for i in endDateC if i != '-']
            
            for i in zip(startDateC, endDateC):
                timeStamps.append(str(i))
            
            df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL'])
            
        dataMatrix = flatter(dataMatrix)
        
    elif span == 'd':
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
        
        if df.empty == True:
            pass
        elif df.empty == False:
            variable = df[varName].values.tolist()
            startDateC = df['startDate'].values.tolist()
            endDateC = df['endDate'].values.tolist()
        
            dataMatrix = [[] for i in range(len(variable))] # With toMatrix() this is not needed
            
            # group variable in nested lists of 7 items
            dataMatrix = toMatrix(variable, 7)

        startDateC = [i for i in startDateC if i != '-']
        endDateC = [i for i in endDateC if i != '-']
        
        for i in zip(startDateC, endDateC):
            timeStamps.append(str(i))

df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL']) 
varGraph = 'mean'

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
    # span = input('All months (a), all months in one or several given years (b), a specific month in every year (c), several months in every year (d) several months in several years (e) or a range of months (f): ')
    span = 'a'
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
    # span = input('All weeks (a), 1st/2nd/3rd/4th week of each month (b), range of weeks in several or all years (c) or range of weeks (d): ')
    span = 'c'
    if span == 'a':
        df = df.loc[df['week'] != 0]
        
    elif span == 'b':
        weekNumber = int(input('Enter the week number: '))
        df = df.loc[df['weekOrder'] == weekNumber]
    
    elif span == 'c':
        # yearBeginOri, monthBegin, dayBegin = input('Enter the first year desired: '), input('Enter the first month desired: '), input('Enter the first day desired: ')
        # numberYears, numberMonths = int(input('Enter number of years to analyze: ')), int(input('Enter the number of months to analyze: '))
        yearBeginOri, monthBegin, dayBegin = '2014', '1', '1'
        numberYears, numberMonths = 8, 4
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


if __name__ == '__main__':

    functionalData, gridPoints, dataMatrix = dataGrid(dataMatrix, timeFrame)

    smoothedData = smoothing(functionalData, dataMatrix)

    smoothedDataGrid, depthInt, depthMBD = depthCalc(smoothedData, gridPoints)
    
    outliers, outliersBoosted = msplot(smoothedData, smoothedDataGrid, integratedDepth, cutoffMDBMS)
    
    
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
    
    result = {'all_values': x,
            'rule_1': [],
            'rule_2': [],
            'rule_3': [],
            'rule_4': [],
            'rule_5': [],
            'rule_6': [],
            'rule_7': [],
            'rule_8': [],
            'fda': outliersBoosted,
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
    
    ax.set_title(r'$O3$ '+r'$\bar{x}$'+f' chart')
    ax.set(xlabel='Date', ylabel=r'$\bar{x}' + r' (\mu*g/m^3)$')
    
    plt.plot(dates, result['all_values'], color='red', markevery=format_arr(1), ls='', marker='s', mfc='none', mec='red', label='Rule1', markersize=mark*5.2)
    plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(2), ls='', marker='o', mfc='none', mec='blue', label="Rule2", markersize=mark*4.7)
    plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(3), ls='', marker='o', mfc='none', mec='purple', label="Rule3", markersize=mark*4.2)
    plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(4), ls='', marker='o', mfc='none', mec='green', label="Rule4", markersize=mark*3.7)
    plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(5), ls='', marker='o', mfc='none', mec='orange', label='Rule5', markersize=mark*3.2)
    plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(6), ls='', marker='o', mfc='none', mec='pink', label='Rule6', markersize=mark*2.7)
    plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(7), ls='', marker='o', mfc='none', mec='cyan', label='Rule7', markersize=mark*2.2)
    plt.plot(dates, result['all_values'], color='blue', markevery=format_arr(8), ls='', marker='o', mfc='none', mec='olive', label='Rule8', markersize=mark*1.7)
    plt.plot(dates, result['all_values'], color='red', marker="o", markersize=mark, label='fda')
    plt.plot(dates, result['all_values'], color='blue', markevery=format_arr_fda(result['fda']), marker="o", markersize=mark)
    plt.xticks(dates, fontsize = 8, rotation = 90)
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
    # rule = str(input('Which Weco rule do you wish to validate (rule_#): '))
    rule = 'rule_5'
    outliers = validator(rule, dates)
    
    print('outliers: ', outliers)

    