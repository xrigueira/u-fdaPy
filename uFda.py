from cv2 import kmeans
from matplotlib.lines import Line2D
import numpy as np
import skfda as fda
import pandas as pd
import plotly.graph_objects as go

from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sympy import zeros
from pearson import pearson_correlation

"""This file implemented the functional data analysis of the pre-processed data"""

# TODO turn the Plotly implementation into a function

def flatter(list):
    return [item for sublits in list for item in sublits]

def labeler(varname):

    if varname == 'Amonio':
        label_title = r'$NH_4$'
        label_y_axis = r'$NH_4$ ' + r'$(m*g/L)$'
    elif varname == 'Conductividad':
        label_title = r'Conductivity'
        label_y_axis = r'Conductivity ' r'$(\mu*S/cm)$'
    elif varname == 'Nitratos':
        label_title = r'$NO_{3^-}$'
        label_y_axis = r'$NO_{3^-}$ ' +r'$(m*g/L)$'
    elif varname == 'Oxigeno disuelto':
        label_title = r'$O_2$'
        label_y_axis = r'$O_2$ ' r'$(m*g/L)$'
    elif varname == 'pH':
        label_title = r'pH'
        label_y_axis = r'pH'
    elif varname == 'Temperatura':
        label_title = r'Temperature'
        label_y_axis = r'Temperature ' +u'(\N{DEGREE SIGN}C)'
    elif varname == 'Caudal':
        label_title = r'Flow'
        label_y_axis = r'Flow ' + r'($m^3/s$)'
    elif varname == "Turbidez":
        label_title = r'Turbidity'
        label_y_axis = r'Turbidity ' + r'(NTU)'
    elif varname == "Pluviometria":
        label_title = r'Pluviometry'
        label_y_axis = r'Pluviometry ' + r'(mm)'



    return label_title, label_y_axis

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

    for nBasis in range(10, 300, 1):

        basis = fda.representation.basis.Fourier(n_basis=nBasis) # fitting the data through Fourier
        smoothedData = functionaldata.to_basis(basis) # Change class to FDataBasis

        evaluatingPoints = smoothedData.evaluate(np.array(gridpoints), derivative=0) # get the corresponding values in the resulting curve
        evaluatingPoints = evaluatingPoints.tolist() # convert from array to list        

        flat2evaluatingPoints = flatter(evaluatingPoints)
        flatevaluatingPoints = flatter(flat2evaluatingPoints)

        # rho, p = pearsonr(np.array(dataMatrixFlat), np.array(flatevaluatingPoints))
        rho = pearson_correlation(dataMatrixFlat, flatevaluatingPoints)

        if rho >= 0.95:
            break
        else:
            continue

    functionaldata.plot()
    smoothedData.plot()
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

    label_title, label_y_axis = labeler(varname=varname)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    funcMSPlot = fda.exploratory.visualization.MagnitudeShapePlot(fdata=smootheddatagrid, multivariate_depth=depth, cutoff_factor=cutoff, axes=ax1)

    outliersMSPlot = funcMSPlot.outliers

    # Copy of the outliers for the control charts
    outliersCC = list(np.copy(outliersMSPlot).astype(int))
    
    funcMSPlot.plot()
    smootheddata.plot(group=funcMSPlot.outliers.astype(int), group_colors=funcMSPlot.colormap([color, outliercolor]), group_names=['No outliers', 'Outliers'], axes=ax2)
    
    # ax2.set_title(f'Outliers {depthName} depth ' + label_title)
    ax2.set_title(f'Functional weekly data ' + label_title)
    ax2.set_xlabel('Time (15min intervals)')
    ax2.set_ylabel(label_y_axis)
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
    if all(outliersMSPlot == 0) == False:
        
        # Take the two dimensions separate
        mag = funcMSPlot.points[:, 0]
        shape = funcMSPlot.points[:, 1]

        # Implement algos.
        from sklearn.cluster import KMeans
        from sklearn.ensemble import IsolationForest
        from sklearn.covariance import MinCovDet
        
        # Isolation Forest
        modeliF = IsolationForest(n_estimators=100, contamination=0.10)
        modeliF.fit(funcMSPlot.points)
        pred = modeliF.predict(funcMSPlot.points)
        probs = -1*modeliF.score_samples(funcMSPlot.points)
        
        fig, axes = plt.subplots(1, figsize=(7, 5))
        sp = axes.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=probs, cmap='rainbow')
        fig.colorbar(sp, label='Simplified Anomaly Score')
        axes.set_title('Isolation Forest Scores ' + label_title)
        axes.set_facecolor("#F1F0E6")
        axes.grid(color='w', linestyle='-', linewidth=1)
        axes.set_axisbelow(True)

        indexiF = np.where(probs >= np.quantile(probs, 0.875))
        valuesiF = funcMSPlot.points[indexiF]

        fig, axes = plt.subplots(1, figsize=(6, 5))
        axes.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1])
        axes.scatter(valuesiF[:, 0], valuesiF[:, 1], color='r')
        axes.set_title("Isolation Forest Binarized " + label_title)
        axes.set_facecolor("#F1F0E6")
        axes.grid(color='w', linestyle='-', linewidth=1)
        axes.set_axisbelow(True)
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='No Outliers', markerfacecolor='b', markersize=13),
                    Line2D([0], [0], marker='o', color='w', label='Outliers', markerfacecolor='r', markersize=13)]
        axes.legend(handles=legend_elements, loc='best')

        # Minimum Covariance Determinant
        modelMinCov = MinCovDet(random_state=0)
        modelMinCov.fit(funcMSPlot.points)
        mahaDistance = modelMinCov.mahalanobis(funcMSPlot.points)

        fig, axes = plt.subplots(1, figsize=(7, 5))
        sp = axes.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=mahaDistance, s=50, cmap='bwr')
        fig.colorbar(sp, label='Mahalanobis Distance')
        axes.set_title("Minimum Covariance Determinant Score " + label_title)
        axes.set_facecolor("#F1F0E6")
        axes.grid(color='w', linestyle='-', linewidth=1)
        axes.set_axisbelow(True)

        indexMinCov = np.where(mahaDistance >= np.quantile(mahaDistance, 0.875))
        valuesMinCov = funcMSPlot.points[indexMinCov]

        fig, axes = plt.subplots(1, figsize=(6, 5))
        axes.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1])
        axes.scatter(valuesMinCov[:, 0], valuesMinCov[:, 1], color='r')
        axes.set_title("Minimum Covariance Determinant Binarized " + label_title)
        axes.set_facecolor("#F1F0E6")
        axes.grid(color='w', linestyle='-', linewidth=1)
        axes.set_axisbelow(True)
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='No Outliers', markerfacecolor='b', markersize=13),
                            Line2D([0], [0], marker='o', color='w', label='Outliers', markerfacecolor='r', markersize=13)]
        axes.legend(handles=legend_elements, loc='best')

        indexiF = [i for i in indexiF[0]]
        indexMinCov = [i for i in indexMinCov[0]]

        # OR option
        indexFinal = indexiF + indexMinCov
        indexFinal = list(dict.fromkeys(indexFinal))

        # AND option
        # indexFinal = [i for i in indexiF if (i in indexiF) and (i in indexMinCov)]

        colors = np.copy(outliersMSPlot).astype(float)
        colors[:] = color
        colors[indexFinal] = outliercolor

        labels = np.copy(funcMSPlot.outliers.astype(int))
        labels[:] = 0
        labels[indexFinal] = 1
        
        # Copy of the labels list for the control charts
        outliersCCBoosted = list(labels.copy())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # ax1 = fig.add_subplot(1, 1, 1)

        ax1.scatter(funcMSPlot.points[:, 0], funcMSPlot.points[:, 1], c=funcMSPlot.colormap(colors))
        ax1.set_title("MS-Plot")
        ax1.set_xlabel("Magnitude outlyingness")
        ax1.set_ylabel("Shape outlyingness")
        # ax1.add_artist(ellipse)
        
        ax2.set_title(f'Functional weekly data ' + label_title)
        ax2.set_xlabel('Time (15min intervals)')
        ax2.set_ylabel(label_y_axis)
        
        colormap = plt.cm.get_cmap('seismic')
        smootheddata.plot(group=labels, group_colors=colormap([color, outliercolor]), group_names=['No outliers', 'Outliers'], axes=ax2)
        
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
        outliersCCBoosted = [0] * len(outliersCC)
    
    # Get the dates of the outleirs
    outliers = [i for i,j in zip(timestamps, outliersMSPlot) if j == 1]

    # print(outIntDepth)
    # print('time stamps:', timeStamps)
    print('outliers:', np.round(len(outliers)/len(timestamps), 3), outliers)
    
    if all(outliersMSPlot == 0) == False:

        outliersBoosted = [i for i,j in zip(timestamps, labels) if j == 1]
        outliersMag = [i for i,j in zip(mag, labels) if j == 1]
        outlierShape = [i for i,j in zip(shape, labels) if j == 1]

        dfOutliers = pd.DataFrame(list(zip(outliersMag, outlierShape)), index=outliersBoosted, columns=['magnitud', 'shape'])
        
        print('outliers boosted:', np.round(len(outliersBoosted)/len(timestamps), 3), outliersBoosted)
        print(dfOutliers)
        print('Average magnitude: {} | Average shape: {}'.format(np.average(mag), np.average(shape)))
    
    else:

        outliersBoosted = [0] * len(outliersMSPlot)
    
    return outliers, outliersBoosted, outliersCC, outliersCCBoosted

def functionalAnalysis(varname, depthname, datamatrix, timestamps, timeframe, depth, cutoff):
    
    gridPoints, functionalData = dataGrid(datamatrix, timeframe)

    smoothedData, smoothedDataGrid = smoothing(datamatrix, gridpoints=gridPoints, functionaldata=functionalData)
    
    outliers, outliersBoosted, outliersCC, outliersCCBoosted = msplot(varname, depthname, timestamps, depth, cutoff, smootheddata=smoothedData, smootheddatagrid=smoothedDataGrid)
    
    # outliers, outliersBoosted, outliersCC, outliersCCBoosted = '', '', '', ''
    
    plt.show()
    
    return outliers, outliersBoosted, outliersCC, outliersCCBoosted