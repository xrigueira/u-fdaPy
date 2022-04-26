
from checkGaps import checkGaps
from normalizer import normalizer
from filterer import filterer
from builder import builder

# Define the data we want to study
variableFile = 'Amonio' # this remains unused for now

if __name__ == '__main__':
    
    # Fill in the gaps in the time series
    checkGaps(File='Amonio.txt')
    print('[INFO] checkGaps() DONE')
    
    # Normalize the data. See normalizer.py for details
    normalizer(File='Amonio_full.csv')
    print('[INFO] normalizer() DONE')
    
    # Filter out those months with too many NaN and iterate on the rest
    filterer(File='Amonio_nor.csv', span='c')
    print('[INFO] filterer() DONE')
    
    # Read the database with the desired time unit and create dataMatrix and timeStamps
    dataMatrix, timeStamps = builder(File='Amonio_pro.csv', timeFrame='c')
    print('[INFO] builder() DONE')