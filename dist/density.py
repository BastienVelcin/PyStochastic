import numpy as np

class distribution:
    def __init__(self,name,support,parameters,density,repartition_function,expectation, variance):
        self.name = name
        self.support = support
        self.parameters = parameters
        self.density = density
        self.repartition_function = repartition_function
        self.expectation = expectation
        self.variance = variance

def dbeta(a,b,x):
    return x**(a-1)*(1-x)**(b-1)