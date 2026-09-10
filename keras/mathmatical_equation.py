#softmax
from math import exp

def softmax(input_vector):
    exponents = [exp(i) for i in input_vector]

    sum_of_exponents = sum(exponents)

    probabilities = [round(i / sum_of_exponents, 3) for i in exponents]

    return probabilities
print(softmax([-3, -2, -1, 0, 1, 2 ,3]))
