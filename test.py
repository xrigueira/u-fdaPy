import skfda as fda
from matplotlib import pyplot as plt

X, y = fda.datasets.fetch_weather(return_X_y=True, as_frame=True)

fd = X.iloc[:, 0].values

print(fd)

msplot = fda.exploratory.visualization.MagnitudeShapePlot(fd, multivariate_depth = fda.exploratory.depth.multivariate.SimplicialDepth())

msplot.plot()

plt.show()