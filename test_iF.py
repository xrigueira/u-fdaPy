import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest

X = np.array([[ 2.57939112e+00,  4.65296104e+00],
[ 2.19984202e+00,  3.02878839e+00],
[ 2.44530614e+00,  2.75930041e+01],
[-3.52702426e-02,  2.86448401e+00],
[-2.40555491e-01,  4.54931409e+00],
[ 2.85520622e+00,  4.80600430e+01],
[ 2.89487337e+00,  1.11783052e+01],
[ 5.95468893e-01,  1.39579666e+00],
[-7.03947258e-01,  1.52233408e+00],
[ 4.35681942e+00,  1.24588306e+01],
[ 2.80423925e-01,  1.77859830e+00],
[-1.04057758e+00,  3.28946709e+00],
[ 6.62774116e-01,  5.14219560e-01],
[-7.15207169e-01,  4.93694940e+00],
[-3.42649010e+00,  7.37429998e+00],
[-8.65472257e-01,  2.19538032e+00],
[ 5.14033714e+00,  1.26187669e+01],
[ 4.99942059e+00,  3.21430624e+01],
[ 4.20729512e+00,  1.10340564e+01],
[ 1.39385457e+01,  6.89942662e+02],
[ 1.90938696e+00,  4.87953760e+00],
[ 2.78366417e+01,  6.70235803e+02],
[ 4.35426524e+00,  8.14189779e+01],
[ 4.53650452e-01,  1.13531224e+00],
[ 6.65529842e-01,  1.74527008e+00],
[ 2.42447553e+00, 1.29281684e+00],
[ 7.00588452e-01,  1.54499232e+00],
[-2.61895640e-02,  3.53682682e+00],
[-1.41969512e+00,  3.28870049e+00],
[-1.49653442e+00,  4.62355891e+01],
[ 1.15188171e+01,  4.03413977e+02],
[-2.29941011e-01,  5.30725106e+00],
[-9.82356386e-01,  4.24370524e+00],
[-3.22955297e+00,  2.62768165e+00],
[-2.95477933e+00,  6.07978023e+00],
[-3.10042905e+00,  2.09318886e+00],
[-3.40967332e+00,  4.50783616e+00],
[-5.62360773e+00,  6.75501171e+00],
[-4.63760490e-01,  2.62279704e+00],
[-3.39926564e-01,  1.96678078e+00],
[ 3.87973961e-01,  1.05705594e+00],
[-1.97733383e+00,  1.84107245e+00],
[-2.66736670e-01,  1.28535800e+00],
[ 4.28605638e-01,  2.26111780e+00],
[-1.94637368e+00,  1.26128955e+00],
[-8.94955404e-01,  4.61765385e-01],
[-3.62951300e-01,  8.60099022e-01],
[-2.29348170e+00,  6.43515192e+00],
[ 8.89874238e+00,  3.66691320e+02],
[ 3.35705828e-01,  1.98192773e+00],
[ 3.13582122e+00,  1.43266293e+00],
[ 1.29032062e+01,  1.03516969e+02],
[ 2.08221563e-01,  1.61899252e+00],
[ 1.75401744e+00,  4.81701398e+00],
[ 3.70597236e+00,  1.08743535e+01],
[-5.98505884e-01,  9.90066927e-01],
[ 1.43372959e+00,  1.43926521e-01],
[-2.78938767e-01,  1.10776108e+00],
[-1.80292195e+00,  1.67054208e+00],
[ 1.51837608e+00,  6.49236042e+00],
[ 1.81368060e+01,  6.06097215e+02],
[-3.98506806e-01,  1.43219343e+00],
[-8.52290226e-01,  2.10337338e+00],
[-9.25486433e+00,  4.31559503e+01],
[-7.97956415e-01,  2.30285714e+00],
[ 2.25996091e+00,  9.58393722e+00],
[ 7.95338235e-01,  2.65182626e+00],
[ 1.72734168e+00,  2.14067937e+00],
[-2.89698430e+00,  2.49194136e+01],
[ 3.50226576e-01,  3.77846022e+00],
[ 3.32634160e+00,  1.77560985e+01],
[ 3.32632613e-01,  8.99339714e-01],
[ 4.08553844e+00,  3.83144391e+00],
[ 7.08019361e+00,  1.68666585e+02],
[-1.22494094e+00,  3.68138066e+01],
[-6.50160270e-01,  2.82145605e+00],
[-1.42945939e+00,  1.22222072e+00],
[ 7.63671968e-01,  2.82332426e+00],
[-1.52466015e+00,  1.31388209e-01],
[-4.29441422e-01,  1.41092493e+00],
[ 2.18531033e+00,  2.96495890e+00],
[ 6.88128423e+00, 1.17599832e+01],
[ 2.59036627e+00,  9.61756792e+00],
[ 1.75049632e-01,  2.72145371e+00],
[ 2.01946728e-01,  3.38376027e+00],
[ 1.80544720e+00,  4.45607112e+00],
[ 1.21930506e+01,  1.79777412e+02],
[ 4.07110541e+00,  1.87940541e+01],
[ 1.12683578e+01,  1.13625128e+02],
[ 5.58648427e-01,  2.37074086e+00],
[-1.39456327e+00,  2.99257306e-02],
[-9.45655183e-01,  2.10891514e+00],
[-4.12390631e+00,  1.26023311e+01],
[ 4.86806305e-01,  1.72982511e+00],
[-2.14501731e-02,  3.59343289e+00],
[-1.37901150e+00,  1.52631206e+00],
[ 4.30338539e+00,  3.48982696e+01],
[-1.43496718e+01,  6.76868677e+02],
[ 1.99626953e+00,  6.94224899e-01],
[ 2.37096789e+00,  3.31869907e+00],
[-5.57656246e-03,  3.56303579e+00],
[ 7.93428211e-01,  1.49483813e+00],
[-6.82181082e-01,  2.60810980e+00],
[-8.74970104e-01,  4.20810461e+00],
[-1.54534764e+00,  8.87774773e-01],
[-9.19847497e-01,  2.34335860e+00],
[-4.92004720e+00, 1.55305442e+01],
[-1.43892286e+01,  1.46504957e+02],
[-9.04778169e+00,  4.58030744e+01],
[-2.41118300e+01,  2.43146252e+02],
[-5.69503394e+00,  2.57256912e+01],
[-2.20267252e+01,  2.93409403e+02],
[-2.48518025e-01,  3.08053704e+01],
[ 4.10085970e+00,  1.69598982e+01],
[-2.25175913e+00,  5.05908203e+00],
[-2.29589446e-01,  4.63689565e+00],
[-1.11731694e+00,  5.25929733e+00],
[ 1.27722708e+00,  5.48155139e+00],
[-2.03430975e+01,  9.09806840e+02],
[-9.04126883e+00,  1.45011576e+02],
[-4.15292250e+00,  2.22627400e+01],
[-2.75013656e-01,  9.83357378e+00],
[-3.00689436e+00,  2.65390917e+00],
[-8.81163124e-01,  1.19806874e+00],
[ 4.29073247e+00,  7.22737078e+01],
[-5.15118260e+00,  3.99457772e+01],
[-1.92345185e+01,  5.86196901e+02],
[ 3.03718323e+00,  4.76522088e+01]])

model = IsolationForest(n_estimators=100, contamination=0.10)
model.fit(X)
pred = model.predict(X)
probs = -1*model.score_samples(X)

# plt.scatter(X[:, 0], X[:, 1], c=pred, cmap='RdBu')

# plt.show()

pred_scores = -1*model.score_samples(X)

plt.scatter(X[:, 0], X[:, 1], c=pred_scores, cmap='RdBu')
plt.colorbar(label='Simplified Anomaly Score')
plt.show()

index = np.where(probs >= np.quantile(probs, 0.875))
values = X[index]

plt.scatter(X[:,0], X[:,1])
plt.scatter(values[:,0], values[:,1], color='r')
plt.show()