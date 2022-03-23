import random
fin = open("wisconsin-rfe-reduced.csv", 'rb')
f75out = open("train.csv", 'wb')
f25out = open("test.csv", 'wb')
for line in fin:
    r = random.random()
    if r < 0.80:
        f75out.write(line)
    else:
        f25out.write(line)
fin.close()
f75out.close()
f25out.close()