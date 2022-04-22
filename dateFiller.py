minute = 00
hour = 00

for i in range(288):
    
    time = f'{hour}:{minute}'
    
    print(time)
    
    if minute == 55:
        hour = hour + 1
        minute = 0
    else:
        minute = minute + 5

