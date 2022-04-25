
import os
import pandas as pd

"""This function acts on the processed database. Its main function
is to select the desired time frames by the user.

Returns:
    - dataMatrix: 2D list with the discrete data ready to be coverted to 
    functional data.
    - timeStamps: the time marks for each time frame."""

File = 'Amonio_pro.csv'
fileName, fileExtension = os.path.splitext(File)
df = pd.read_csv(f'Database/{fileName}.csv', delimiter=';') # Set column date as the index
cols = list(df.columns.values.tolist())

timeFrame = 'a'

if timeFrame == 'a':

    # Get information on the time frame desired by the user
    span = input('All months (a), all months in one or several given years (b), a specific month in every year (c), several months in every year (d) several months in several years (e) or a range of months (f): ')

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

    # Select the data in the specified time frame
    years = list(dict.fromkeys(df['year'].tolist()))
    months = list(dict.fromkeys(df['month'].tolist()))
    months.sort()

    # Initialize two lists for dataMatrix and timeStamps
    dataMatrix = []
    timeStamps =  []

    if span == 'a':

        # Put the desired data in dataMatrix 
        for i in years:

            df = df.loc[df['year'] == i]

            for j in months:

                df = df.loc[df['month'] == j]

                if df.empty == True:
                    pass


elif timeFrame == 'c':

    # Get information on the time frame desired by the user
    span = input('All weeks (a), 1st/2nd/3rd/4th week of each month (b), range of weeks in several or all years (c) or range of weeks (d): ')

    if span == 'b':
        weekNumber = list(map(int, input('Enter the week number: ')))
    
    elif span == 'c':
        yearBeginOri, monthBegin, dayBegin = input('Enter the first year desired: '), input('Enter the first month desired: '), input('Enter the first day desired: ')
        numberYears, numberMonths = int(input('Enter number of years to analyze: ')), int(input('Enter the number of months to analyze: '))
    
    elif span == 'd':
        yearBegin, monthBegin, dayBegin = input('Enter the first year desired: '), input('Enter the first month desired: '), input('Enter the first day desired: ')
        yearEnd, monthEnd, dayEnd = input('Enter the last year desired: '), input('Enter the last month desired: '), input('Enter the last day desired: ')