import numpy as np
from tensors import Tensor

class operations:

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def run(cls, *args, **kwargs):
        obj = cls()
        inputs = tuple(a.data for a in args)
        result = obj.forward(*inputs, **kwargs)
        obj.output = result
        return Tensor(result, parents=args, creator=obj)

class Mul(operations):
    def forward(self, x, y):
        self.x,self.y = x,y
        return self.x*self.y
    def backward(self, gradient):
        return tuple([self.y*gradient, self.x*gradient])
    