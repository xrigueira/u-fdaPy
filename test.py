from numpy import empty
import skfda as fda
from varname import nameof

integratedDepth = fda.exploratory.depth.IntegratedDepth().multivariate_depth
modifiedbandDepth = fda.exploratory.depth.ModifiedBandDepth().multivariate_depth
projectionDepth = fda.exploratory.depth.multivariate.ProjectionDepth()
simplicialDepth = fda.exploratory.depth.multivariate.SimplicialDepth()

integratedDepth

print(nameof(integratedDepth)[0:-5])

