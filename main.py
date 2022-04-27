import skfda as fda

from sklearn import preprocessing
from checkGaps import checkGaps
from normalizer import normalizer
from filterer import filterer
from builder import builder
from uFda import functionalAnalysis

# Define the data we want to study
varName = 'Amonio' # this remains unused for now

# Set the preprocessing option
preprocessing = 'Y'

if __name__ == '__main__':
    
    if preprocessing == 'Y':
        
        # Fill in the gaps in the time series
        checkGaps(File='Amonio.txt')
        print('[INFO] checkGaps() DONE')
        
        # Normalize the data. See normalizer.py for details
        normalizer(File='Amonio_full.csv')
        print('[INFO] normalizer() DONE')
        
        # Filter out those months with too many NaN and iterate on the rest
        filterer(File='Amonio_nor.csv', span='a')
        print('[INFO] filterer() DONE')
    
    else:
        
        # Read the database with the desired time unit and create dataMatrix and timeStamps
        dataMatrix, timeStamps = builder(File='Amonio_pro.csv', timeFrame='a')
        print('[INFO] builder() DONE')
        
        # Initialize needed variables
        cutoffIntBox, cutoffMDBBox, cutoffIntMS, cutoffMDBMS = 1, 1, 0.993, 0.993 # Cutoff params

        # Define depths here
        integratedDepth = fda.exploratory.depth.IntegratedDepth().multivariate_depth
        modifiedbandDepth = fda.exploratory.depth.ModifiedBandDepth().multivariate_depth
        projectionDepth = fda.exploratory.depth.multivariate.ProjectionDepth()
        simplicialDepth = fda.exploratory.depth.multivariate.SimplicialDepth()
        
        functionalAnalysis(varname=varName, datamatrix=dataMatrix, timestamps=timeStamps, timeframe='a', depth=modifiedbandDepth, cutoff=cutoffIntMS)
        print('[INFO] functionalAnalysis() DONE')