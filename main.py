
from checkGaps import checkGaps
from normalizer import normalizer

if __name__ == '__main__':
    
    # Fill in the gaps in the time series
    checkGaps('Amonio.txt')
    
    # Normalize the data. See normalizer.py for details.
    normalizer('Amonio_full.txt')
    