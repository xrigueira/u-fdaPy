import scipy


import numpy as np
from scipy.stats.stats import pearsonr
from pearson import pearson_correlation
datamatrixFlat = [1.1, 2.3, 2.9, 3.4, 5.0]
flatevaluatingPoints = [1, 2, 3, 4, 5]

rho, p = pearsonr(np.array(datamatrixFlat), np.array(flatevaluatingPoints))

print(rho, p)

rho = pearson_correlation(datamatrixFlat, flatevaluatingPoints)

print(rho)