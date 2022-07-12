
import skfda as fda

from checkGaps import checkGaps
from normalizer import normalizer
from filterer import filterer
from builder import builder
from uFda import functionalAnalysis
from controlCharts import controlCharts

"""This file performs the univariate functional data analysis
with directional outlyingness"""

# Define the data we want to study
varName = 'Pluviometria'
timeFrame = 'b'

# Set the preprocessing option
preprocessing = 'n'

if __name__ == '__main__':
    
    # Perform the univariate preprocessing
    if preprocessing == 'Y':
        
        # Fill in the gaps in the time series
        checkGaps(File=f'{varName}.txt')
        print('[INFO] checkGaps() DONE')
        
        # Normalize the data. See normalizer.py for details
        normalizer(File=f'{varName}_full.csv')
        print('[INFO] normalizer() DONE')
        
        # Filter out those time units with too many NaN and iterate on the rest
        filterer(File=f'{varName}_nor.csv', timeframe=timeFrame)
        print('[INFO] filterer() DONE')
    
    else:
        
        # Read the database with the desired time unit and create dataMatrix and timeStamps
        dataMatrix, timeStamps = builder(File=f'{varName}_pro.csv', timeFrame=timeFrame)
        print('[INFO] builder() DONE')

        cutoffIntBox, cutoffMDBBox, cutoffIntMS, cutoffMDBMS = 1, 1, 0.993, 0.993 # Cutoff params

        # Define depths here
        integratedDepth = fda.exploratory.depth.IntegratedDepth().multivariate_depth
        modifiedbandDepth = fda.exploratory.depth.ModifiedBandDepth().multivariate_depth
        projectionDepth = fda.exploratory.depth.multivariate.ProjectionDepth()
        simplicialDepth = fda.exploratory.depth.multivariate.SimplicialDepth()
        
        outliers, outliersBoosted, outliersCC, outliersCCBoosted = functionalAnalysis(varname=varName, depthname='modified band', datamatrix=dataMatrix, timestamps=timeStamps, timeframe=timeFrame, depth=modifiedbandDepth, cutoff=cutoffIntMS)
        print('[INFO] functionalAnalysis() DONE')

        controlCharts(varname=varName, datamatrix=dataMatrix, timestamps=timeStamps, timeframe=timeFrame, vargraph='mean', outleirsresults=outliersCCBoosted)
        print('[INFO] controlCharts() DONE')
