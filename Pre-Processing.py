import pandas
data = pandas.read_csv("wisconsin.csv", names=['a','b','c','d','e','f','g','h','i','j','k'])
data['k'].replace('2',0, inplace = True)
data['k'].replace('4',1, inplace = True)
del data['a']
data.to_csv("wisconsin-index-column-removed.csv", index=None, header=None)
print(data)
