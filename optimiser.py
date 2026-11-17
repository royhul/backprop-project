import numpy as np

class SGD:
    def __init__(self, parameters, lr):
        self.lr = lr
        self.parameters = parameters
    def zero_grad(self):
        for ele in self.parameters:
            ele.grad = np.zeros_like(ele.data)
    def step(self):
        for ele in self.parameters:
            ele.data -= self.lr * ele.grad