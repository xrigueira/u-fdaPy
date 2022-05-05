from cv2 import kmeans
import numpy as np
import skfda as fda
import pandas as pd
import plotly.graph_objects as go

from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr

"""This file implemented the functional data analysis of the pre-processed data"""

# TODO turn the Plotly implementation into a function

def flatter(list):
    return [item for sublits in list for item in sublits]

def dataGrid(datamatrix, timeframe):
    
    # Define object FDataGrid
    if timeframe == 'a':
        gridPoints = list(range(2976))
    elif timeframe == 'b':
        gridPoints = list(range(672))
    elif timeframe == 'c':
        gridPoints = list(range(96))
    
    functionalData = fda.FDataGrid(data_matrix=datamatrix, grid_points=gridPoints)
    
    # Plotting the data
    # fig, axes = plt.subplots()
    
    # functionalData.plot(axes=axes)
    
    # axes.set_title(f'Original data {varName}')
    # axes.set_xlabel('Days')
    # axes.set_ylabel(f'{varName}')
    # fig.savefig(f'original_{varName}.png')
    
    return gridPoints, functionalData

def smoothing(datamatrix, gridpoints, functionaldata):
    
    # Calculate Fourier smoothing and the number of basis functions
    dataMatrixFlat = flatter(datamatrix)
    
    for nBasis in range(300):
        
        basis = fda.representation.basis.Fourier(n_basis=nBasis) # fitting the data through Fourier
        smoothedData = functionaldata.to_basis(basis) # Change class to FDataBasis
        
        evaluatingPoints = smoothedData.evaluate(np.array(gridpoints), derivative=0) # get the corresponding values in the resulting curve
        evaluatingPoints = evaluatingPoints.tolist() # convert from array to list

        flat2evaluatingPoints = flatter(evaluatingPoints)
        flatevaluatingPoints = flatter(flat2evaluatingPoints)
    
        rho, p = pearsonr(np.array(dataMatrixFlat), np.array(flatevaluatingPoints))
        
        if rho >= 0.95 or nBasis >= len(datamatrix[0]):
            break
        else:
            continue

    print('Number of basis functions: ', nBasis, 'and rho: ', rho)
    
    # Plotting of the smoothed functions
    # fig, axes = plt.subplots()
    
    # smoothedData.plot(axes=axes)
    # axes.set_title(f'Smoothed data {varName}')
    # axes.set_xlabel('Days')
    # axes.set_ylabel(f'{varName}')
    # fig.savefig(f'smoothed_{varName}.png')
    
    smoothedDataGrid = smoothedData.to_grid(grid_points=gridpoints) # Convert to FDataGrid for further needs
    
    return smoothedData, smoothedDataGrid

def boxplot(varname, depthname, timestamps, depth, cutoff, smootheddata, smootheddatagrid):
    
    # Plot the outliers (Boxplot)
    boxplot = fda.exploratory.visualization.Boxplot(smootheddatagrid, depth_method=depth, factor=cutoff)
    boxplot.show_full_outliers = True
    outliersBoxplot = boxplot.outliers
    
    # boxplot.plot()
    
    # Simplified representation
    color, outliercolor = 0.3, 0.7
    depthName = depthname

    fig, axes = plt.subplots()
    
    smootheddata.plot(group=boxplot.outliers.astype(int), group_colors=boxplot.colormap([color, outliercolor]), group_names=['No outliers', 'Outliers'], axes=axes)
    axes.set_title(f'Outliers {depthName} boxplot {varname}')
    axes.set_xlabel('Days')
    axes.set_ylabel(f'{varname}')
    # fig.savefig(f'results/boxplot/outliers_Integrated_Boxplot_{varName}.png')
    # plt.show()
    
    # Plotly implementation to display the results on the browser
    dataPly = []
    for i in (smootheddatagrid.data_matrix).tolist():
        dataPly.append(flatter(i))
    
    dfPlotly = pd.DataFrame.from_records(dataPly)
    dfPlotly = dfPlotly.transpose()
    dfPlotly.columns = timestamps
    
    outliersBoxplotP = list(outliersBoxplot)
    
    for i, j in enumerate(outliersBoxplotP):
        if j == True:
            outliersBoxplotP[i] = 'red'
        elif j == False:
            outliersBoxplotP[i] = 'blue'
    
    colorDict = {}
    for i, j in zip(timestamps, outliersBoxplotP):
        colorDict.update({i: j})
    
    fig = go.Figure()
    
    for col in dfPlotly.columns:
        fig.add_trace(go.Scatter(x=dfPlotly.index, y=dfPlotly[col], mode='lines', name=col, marker_color=colorDict[col]))
    
    # fig.show()
    
    # Get the dates of the outleirs
    outliers = [i for i,j in zip(timestamps, outliersBoxplot) if j == 1]
    
    # print(outIntDepth)
    # print('time stamps:', timeStamps)
    print('outliers boxplot:', np.round(len(outliers)/len(timestamps), 3), outliers)   
        
    return outliers

def msplot(varname, depthname, timestamps, depth, cutoff, smootheddata, smootheddatagrid):
    
    color, outliercolor = 0.3, 0.7
    depthName = depthname
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    
    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smootheddatagrid, multivariate_depth=depth, cutoff_factor=cutoff, axes=ax1)

    outliersMSPlot = funcMSPlot.outliers
    
    # Copy of the outliers for the control charts
    outliersCC = list(np.copy(outliersMSPlot).astype(int))
    
    funcMSPlot.plot()
    smootheddata.plot(group=funcMSPlot.outliers.astype(int), group_colors=funcMSPlot.colormap([color, outliercolor]), group_names=['nonoutliers', 'outliers'], axes=ax2)
    
    ax2.set_title(f'Outliers {depthName} depth ' + r'$O3$')
    ax2.set_xlabel('Days')
    ax2.set_ylabel(r'$O3$' + r' $(\mu*g/m^3)$')
    # fig.savefig(f'outliers_MSPlot_{varName}.png')
    # plt.show()
    
    # Plotly implementation to display the results on the browser
    dataPly = []
    for i in (smootheddatagrid.data_matrix).tolist():
        dataPly.append(flatter(i))
    
    dfPlotly = pd.DataFrame.from_records(dataPly)
    dfPlotly = dfPlotly.transpose()
    dfPlotly.columns = timestamps 
    
    # Create color dictionary
    outliersMSPlotP = list(outliersMSPlot)
    for i, j in enumerate(outliersMSPlotP):
        if j == True:
            outliersMSPlotP[i] = 'red'
        elif j == False:
            outliersMSPlotP[i] = 'blue'

    colorDict = {}
    for i, j in zip(timestamps, outliersMSPlotP):
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

        # Implement k-Means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(funcMSPlot.points)
        labels = kmeans.predict(funcMSPlot.points)
        index_labels = [i for i, e in enumerate(labels) if e == 1]
        # # Implementacion elipse
        # b = np.percentile(shape, 85)
        # a = np.percentile(mag, 90) - np.percentile(mag, 10)
        # elip = np.where(1 < (((pow((mag), 2) // pow(a, 2)) + (pow((shape), 2) // pow(b, 2)))))
        # # ellipse = Ellipse((0, 0), (a), (b), angle=0, alpha=0.3)

        colors = np.copy(outliersMSPlot).astype(float)
        colors[:] = color
        colors[index_labels] = outliercolor

        labels = np.copy(funcMSPlot.outliers.astype(int))
        labels[:] = 0
        labels[index_labels] = 1
        
        # Copy of the labels list for the control charts
        outliersCCBoosted = list(labels.copy())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        # ax1 = fig.add_subplot(1, 1, 1)

        ax1.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=funcMSPlot.colormap(colors))
        ax1.set_title("MS-Plot")
        ax1.set_xlabel("Magnitude outlyingness")
        ax1.set_ylabel("Shape outlyingness")
        # ax1.add_artist(ellipse)
        
        ax2.set_title(f'Outliers {depthName} depth ' + r'$NH_4$')
        ax2.set_xlabel('data points')
        ax2.set_ylabel(r'$NH_4$' + r' $(mg/l)$')
        
        colormap = plt.cm.get_cmap('seismic')
        smootheddata.plot(group=labels, group_colors=colormap([color, outliercolor]), group_names=['nonoutliers', 'outliers'], axes=ax2)
        
        # Plotly implementation to display the results on the browser
        dataPly = []
        for i in (smootheddatagrid.data_matrix).tolist():
            dataPly.append(flatter(i))
        
        dfPlotly = pd.DataFrame.from_records(dataPly)
        dfPlotly = dfPlotly.transpose()
        dfPlotly.columns = timestamps
        
        # Create color dictionary
        labelsPlotly = list(labels)
        for i, j in enumerate(labels):
            if j == 1:
                labelsPlotly[i] = 'red'
            elif j == 0:
                labelsPlotly[i] = 'blue'

        colorDict = {}
        for i, j in zip(timestamps, labelsPlotly):
            colorDict.update({i: j})
        
        # Graph the results
        fig = go.Figure()
        
        for col in dfPlotly.columns:
            fig.add_trace(go.Scatter(x=dfPlotly.index, y=dfPlotly[col], mode='lines', name=col, marker_color=colorDict[col]))
        
        # fig.show()
    
    else:
        labels = []
    
    # Get the dates of the outleirs
    outliers = [i for i,j in zip(timestamps, outliersMSPlot) if j == 1]

    # print(outIntDepth)
    # print('time stamps:', timeStamps)
    print('outliers:', np.round(len(outliers)/len(timestamps), 3), outliers)
    
    outliersBoosted = [i for i,j in zip(timestamps, labels) if j == 1]
    outliersMag = [i for i,j in zip(mag, labels) if j == 1]
    outlierShape = [i for i,j in zip(shape, labels) if j == 1]

    dfOutliers = pd.DataFrame(list(zip(outliersMag, outlierShape)), index=outliersBoosted, columns=['magnitud', 'shape'])
    
    print('outliers boosted:', np.round(len(outliersBoosted)/len(timestamps), 3), outliersBoosted)
    print(dfOutliers)
    print('Average magnitude: {} | Average shape: {}'.format(np.average(mag), np.average(shape)))
    
    return outliers, outliersBoosted, outliersCC, outliersCCBoosted

def functionalAnalysis(varname, depthname, datamatrix, timestamps, timeframe, depth, cutoff):
    
    gridPoints, functionalData = dataGrid(datamatrix, timeframe)
    
    smoothedData, smoothedDataGrid = smoothing(datamatrix, gridpoints=gridPoints, functionaldata=functionalData)
    
    outliers, outliersBoosted, outliersCC, outliersCCBoosted = msplot(varname, depthname, timestamps, depth, cutoff, smootheddata=smoothedData, smootheddatagrid=smoothedDataGrid)
    
    plt.show()
    
    return outliers, outliersBoosted, outliersCC, outliersCCBoosted