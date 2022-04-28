import numpy as np
import pandas as pd

df = pd.DataFrame({'a':[1,2,np.NaN, np.NaN, np.NaN, 6,7,8,9,10,np.NaN,np.NaN,13,14]})

consecutive = max(df['a'].isnull().astype(int).groupby(df.a.notnull().astype(int).cumsum()).sum())

print(type(consecutive))



