def xxx():
    for i in range(10):
        yield i

    for j in range(19,30):
        yield j


for x in xxx():
    print(x)