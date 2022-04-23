import numpy as np

yearInit, monthInit, dayInit, hourInit, minInit, secInit = 2019, 2, 28 + 1, 0, 0, 0
rows = [[f'{yearInit}-{monthInit}-{dayInit} 00:00:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit]]

for j in range(287):
    
    minInit += 15

    if hourInit == 23 and minInit > 45:
        
        minInit = 0
        hourInit = 0
        dayInit = dayInit + 1
        
    if minInit > 45:

        minInit = 0
        hourInit = hourInit + 1
    
    rows.append([f'{yearInit}-{monthInit}-{dayInit} {hourInit}:{minInit}:00', np.nan, yearInit, monthInit, dayInit, hourInit, minInit, secInit])

print(rows)
print(len(rows))