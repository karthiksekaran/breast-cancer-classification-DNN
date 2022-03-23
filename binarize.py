import pandas
data = pandas.read_csv("25507-100.csv").values
from sklearn.preprocessing import normalize
bin = normalize(data)
import numpy as np
np.savetxt("binarized-25507-100.csv", bin, fmt="%s", delimiter=",")
