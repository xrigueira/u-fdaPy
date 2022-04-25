
import os
import numpy as np
import pandas as pd
import skfda as fda
import plotly.graph_objects as go
from math import pi
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
from skfda.exploratory import outliers
from matplotlib.patches import Ellipse

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

def boxplot(smoothedData, smoothedDataGrid, depth, cutoff):
    
    # Plot the outliers (Boxplot)
    boxplot = fda.exploratory.visualization.Boxplot(smoothedDataGrid, depth_method=depth, factor=cutoff)
    boxplot.show_full_outliers = True
    outliersBoxplot = boxplot.outliers
    
    # boxplot.plot()
    
    # Simplified representation
    color, outliercolor = 0.3, 0.7
    
    if depth == integratedDepth: 
        depthName = 'integrated'
    elif depth == modifiedbandDepth:
        depthName = 'modified band'
    elif depth == projectionDepth:
        depthName = 'projection'
    else:
        depthName = 'simplicial'

    fig, axes = plt.subplots()
    
    smoothedData.plot(group=boxplot.outliers.astype(int), group_colors=boxplot.colormap([color, outliercolor]), group_names=['No outliers', 'Outliers'], axes=axes)
    axes.set_title(f'Outliers {depthName} boxplot {varName}')
    axes.set_xlabel('Days')
    axes.set_ylabel(f'{varName}')
    # fig.savefig(f'results/boxplot/outliers_Integrated_Boxplot_{varName}.png')
    # plt.show()
    
    dataPly = []
    for i in (smoothedDataGrid.data_matrix).tolist():
        dataPly.append(flatter(i))
    
    dfPlotly = pd.DataFrame.from_records(dataPly)
    dfPlotly = dfPlotly.transpose()
    dfPlotly.columns = timeStamps
    
    outliersBoxplotP = list(outliersBoxplot)
    
    for i, j in enumerate(outliersBoxplotP):
        if j == True:
            outliersBoxplotP[i] = 'red'
        elif j == False:
            outliersBoxplotP[i] = 'blue'
    
    colorDict = {}
    for i, j in zip(timeStamps, outliersBoxplotP):
        colorDict.update({i: j})
    
    fig = go.Figure()
    
    for col in dfPlotly.columns:
        fig.add_trace(go.Scatter(x=dfPlotly.index, y=dfPlotly[col], mode='lines', name=col, marker_color=colorDict[col]))
    
    fig.show()
    
    if timeFrame == 'a':
        
        outliers = [i for i,j in zip(years, outliersBoxplot) if j == 1]
        
        # print(outIntDepth)
        # print('time stamps:', timeStamps)
        print('outliers boxplot:', np.round(len(outliers)/len(timeStamps), 3), outliers)
    
    elif timeFrame == 'b' or timeFrame == 'c':
        
        outliers = [i for i,j in zip(timeStamps, outliersBoxplot) if j == 1]
        
        # print(outIntDepth)
        # print('time stamps:', timeStamps)
        print('outliers boxplot:', np.round(len(outliers)/len(timeStamps), 3), outliers)   
        
    return outliers
    
def msplot(smoothedData, smoothedDataGrid, depth, cutoff):
    
    color, outliercolor = 0.3, 0.7
    
    if depth == integratedDepth: 
        depthName = 'integrated'
    elif depth == modifiedbandDepth:
        depthName = 'modified band'
    elif depth == projectionDepth:
        depthName = 'projection'
    else:
        depthName = 'simplicial'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smoothedDataGrid, multivariate_depth=depth, cutoff_factor=cutoff, axes=ax1)

    outliersMSPlot = funcMSPlot.outliers
    
    funcMSPlot.plot()
    smoothedData.plot(group=funcMSPlot.outliers.astype(int), group_colors=funcMSPlot.colormap([color, outliercolor]), group_names=['nonoutliers', 'outliers'], axes=ax2)
    
    ax2.set_title(f'Outliers {depthName} depth ' + r'$O3$')
    ax2.set_xlabel('Days')
    ax2.set_ylabel(r'$O3$' + r' $(\mu*g/m^3)$')
    # fig.savefig(f'outliers_MSPlot_{varName}.png')
    # plt.show()
    
    # Display the results on the browser with Plotly
    dataPly = []
    for i in (smoothedDataGrid.data_matrix).tolist():
        dataPly.append(flatter(i))
    
    dfPlotly = pd.DataFrame.from_records(dataPly)
    dfPlotly = dfPlotly.transpose()
    dfPlotly.columns = timeStamps 
    
    # Create color dictionary
    outliersMSPlotP = list(outliersMSPlot)
    for i, j in enumerate(outliersMSPlotP):
        if j == True:
            outliersMSPlotP[i] = 'red'
        elif j == False:
            outliersMSPlotP[i] = 'blue'

    colorDict = {}
    for i, j in zip(timeStamps, outliersMSPlotP):
        colorDict.update({i: j})
    
    # Graph the results
    fig = go.Figure()
    
    for col in dfPlotly.columns:
        fig.add_trace(go.Scatter(x=dfPlotly.index, y=dfPlotly[col], mode='lines', name=col, marker_color=colorDict[col]))
    
    # fig.show()
    
    # Double filter
    if len(outliersMSPlot) != 0:
        
        # Take the two dimensions separate
        mag = funcMSPlot.points[:, 0]
        shape = funcMSPlot.points[:, 1]

        mag90 = np.where((mag < np.percentile(mag, 7)) | (mag > np.percentile(mag, 93)))
        shape90 = np.where(shape > np.percentile(shape, 95))

        # Implementacion elipse
        b = np.percentile(shape, 85)
        a = np.percentile(mag, 80) - np.percentile(mag, 20)
        elip = np.where(1 < (((pow((mag), 2) // pow(a, 2)) + (pow((shape), 2) // pow(b, 2)))))
        # ellipse = Ellipse((0, 0), (a), (b), angle=0, alpha=0.3)

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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        # ax1 = fig.add_subplot(1, 1, 1)

        ax1.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=funcMSPlot.colormap(colors))
        ax1.set_title("MS-Plot")
        ax1.set_xlabel("Magnitude outlyingness")
        ax1.set_ylabel("Shape outlyingness")
        # ax1.add_artist(ellipse)
        
        ax2.set_title(f'Outliers {depthName} depth ' + r'$NO2$')
        ax2.set_xlabel('Days')
        ax2.set_ylabel(r'$NO2$' + r' $(\mu*g/m^3)$')
        
        colormap = plt.cm.get_cmap('seismic')
        smoothedData.plot(group=labels, group_colors=colormap([color, outliercolor]), group_names=['nonoutliers', 'outliers'], axes=ax2)

        # Display the new results on the browser with Plotly
        dataPly = []
        for i in (smoothedDataGrid.data_matrix).tolist():
            dataPly.append(flatter(i))
        
        dfPlotly = pd.DataFrame.from_records(dataPly)
        dfPlotly = dfPlotly.transpose()
        dfPlotly.columns = timeStamps
        
        # Create color dictionary
        labelsPlotly = list(labels)
        for i, j in enumerate(labels):
            if j == 1:
                labelsPlotly[i] = 'red'
            elif j == 0:
                labelsPlotly[i] = 'blue'

        colorDict = {}
        for i, j in zip(timeStamps, labelsPlotly):
            colorDict.update({i: j})
        
        # Graph the results
        fig = go.Figure()
        
        for col in dfPlotly.columns:
            fig.add_trace(go.Scatter(x=dfPlotly.index, y=dfPlotly[col], mode='lines', name=col, marker_color=colorDict[col]))
        
        # fig.show()
    
    else:
        labels = []
    
    # Output outliers as dates
    if timeFrame == 'a':
        
        outliers = [i for i,j in zip(years, outliersMSPlot) if j == 1]
        
        # print(outIntDepth)
        # print('time stamps:', timeStamps)
        print('outliers:', np.round(len(outliers)/len(timeStamps), 3), outliers)
        
        outliersBoosted = [i for i,j in zip(years, labels) if j == 1]
        outliersMag = [i for i,j in zip(mag, labels) if j == 1]
        outlierShape = [i for i,j in zip(shape, labels) if j == 1]

        dfOutliers = pd.DataFrame(list(zip(outliersMag, outlierShape)), index=outliersBoosted, columns=['magnitud', 'shape'])
        
        print('outliers test boosted:', np.round(len(outliersBoosted)/len(timeStamps), 3), outliersBoosted)
        print(dfOutliers)
        print('Average magnitude: {} | Average shape: {}'.format(np.average(mag), np.average(shape)))
        
    elif timeFrame == 'b':
        
        outliers = [i for i,j in zip(timeStamps, outliersMSPlot) if j == 1]

        # print(outIntDepth)
        # print('time stamps:', timeStamps)
        print('outliers:', np.round(len(outliers)/len(timeStamps), 3), outliers)
        
        outliersBoosted = [i for i,j in zip(timeStamps, labels) if j == 1]
        outliersMag = [i for i,j in zip(mag, labels) if j == 1]
        outlierShape = [i for i,j in zip(shape, labels) if j == 1]

        dfOutliers = pd.DataFrame(list(zip(outliersMag, outlierShape)), index=outliersBoosted, columns=['magnitud', 'shape'])
        
        print('outliers boosted:', np.round(len(outliersBoosted)/len(timeStamps), 3), outliersBoosted)
        print(dfOutliers)
        print('Average magnitude: {} | Average shape: {}'.format(np.average(mag), np.average(shape)))

    elif timeFrame == 'c':
        
        outliers = [i for i,j in zip(timeStamps, outliersMSPlot) if j == 1]
        
        # print(outIntDepth)
        # print('time stamps:', timeStamps)
        print('outliers', np.round(len(outliers)/len(timeStamps), 3), outliers)

        outliersBoosted = [i for i,j in zip(timeStamps, labels) if j == 1]
        outliersMag = [i for i,j in zip(mag, labels) if j == 1]
        outlierShape = [i for i,j in zip(shape, labels) if j == 1]

        dfOutliers = pd.DataFrame(list(zip(outliersMag, outlierShape)), index=outliersBoosted, columns=['magnitud', 'shape'])
        
        print('outliers test boosted:', np.round(len(outliersBoosted)/len(timeStamps), 3), outliersBoosted)
        print(dfOutliers)
        print('Average magnitude: {} | Average shape: {}'.format(np.average(mag), np.average(shape)))
    
    return outliers, outliersBoosted

# File selection
file = 'argentina2_pro.csv'
fileName, fileExtension = os.path.splitext(f'DataBase\{file}')

# Load the data frame and get the columns
df = pd.read_csv(f'DataBase/{file}', delimiter=';', index_col=['date'], parse_dates=['date'], na_values=['NULL']) 
cols = list(df.columns.values.tolist())
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
varName = 'o3'
# timeFrame = input('Years (a), months (b) or weeks (c): ')
timeFrame = 'c'
if timeFrame == 'a':
    span = input('Use a range of years (Y/n): ') 
    
    if span == 'Y':
        numberYears = list(map(int, input('Enter the years within the range desired (space-separated): ').split()))
    else:
        pass

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


if timeFrame == 'a':
    
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
        

if __name__ == '__main__':

    functionalData, gridPoints, dataMatrix = dataGrid(dataMatrix, timeFrame)

    smoothedData = smoothing(functionalData, dataMatrix)

    smoothedDataGrid, depthInt, depthMBD = depthCalc(smoothedData, gridPoints)

    # outliers = boxplot(smoothedData, smoothedDataGrid, integratedDepth, cutoffIntBox)
    
    outliers, outliersBoosted = msplot(smoothedData, smoothedDataGrid, integratedDepth, cutoffMDBMS)
    
    # Display results
    plt.show()

    print('DONE')