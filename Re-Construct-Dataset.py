import pandas
data = pandas.read_csv("wisconsin-index-column-removed.csv", names=['a','b','c','d','e','f','g','h','i','j'])
del data['a']
del data['b']
del data['d']
del data['e']
del data['h']
data.to_csv("wisconsin-rfe-reduced.csv", index=None, header=None)
print(data)
# delete b,f