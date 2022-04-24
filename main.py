
from checkGaps import checkGaps
from normalizer import normalizer
from filterer import filterer

if __name__ == '__main__':
    
    # Fill in the gaps in the time series
    checkGaps(File='Amonio.txt')
    
    # Normalize the data. See normalizer.py for details
    normalizer(File='Amonio_full.txt')
    
    # Filter out those months with too many NaN and iterate on the rest
    filterer(File='Amonio_nor.csv', span='a')
    