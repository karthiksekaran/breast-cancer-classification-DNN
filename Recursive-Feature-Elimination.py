from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
dt = "wisconsin-index-column-removed.csv"
names = ['clump', 'cell-size', 'cell-shape', 'adhesion', 'epithelial', 'bare-nuclei', 'bland', 'nucleoli','mitoses', 'Class']
dataframe = read_csv(dt, names=names)
array = dataframe.values
X = array[:,0:9]
Y = array[:,9]
model = LogisticRegression();
rfe = RFE(model, 4)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

# c,f,g,i