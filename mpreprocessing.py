import os

from checkGaps import checkGaps
from normalizer import normalizer
from joiner import joiner
from mfilterer import mfilterer
from splitter import splitter

"""This file performs the multivariate preprocessing. In order to be able
to compare each variable and validate the outleirs we need to make sure that
each variable has the same lenght. This is not a critical contidion in the
univariate analysis. For that reason, if I want to compare the relation
between different variables I have to preprocess them together making sure
those time frames with too many gaps will be eliminated in all variables and not
just one (which is the univariate case). This way we ensure equality of length 
for each variable, what makes the comparable."""

# Define the data we want to study
files = [f for f in os.listdir("Database") if os.path.isfile(os.path.join("Database", f))]

varNames = [i[0:-4] for i in files] # extract the names of the variables

# Define the time dram we want to use (a: months, b: weeks, c: days)
timeFrame = 'b'


if __name__ == '__main__':

    for varName in varNames:

        # Fill in the gaps in the time series
        checkGaps(File=f'{varName}.txt')
        print('[INFO] checkGaps() DONE')
        
        # Normalize the data. See normalizer.py for details
        normalizer(File=f'{varName}_full.csv')
        print('[INFO] normalizer() DONE')
    
    # Join the normalized databases
    joiner()
    print('[INFO] joiner() DONE')

    # Filter out those months or weeks or days (depending on the desired
    # time unit) with too many NaN in several variables and iterate on the rest
    mfilterer(File='data_joi.csv', timeframe=timeFrame)
    print('[INFO] filterer() DONE')

    # Split each variable into its own separate processed file
    splitter(File="data_pro.csv")
    print('[INFO] splitter() DONE')