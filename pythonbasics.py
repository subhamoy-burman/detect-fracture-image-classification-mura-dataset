from functools import reduce

my_list = [5,7,8,9,11]

def doubler (x):
    return 2*x

output = map(doubler,my_list)

print(list(output))

output2 = map(lambda x:2*x,my_list)

print(list(output2))

output3 = reduce(lambda x,y: x+y, my_list)

print(output3)



