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
        args = list(args)
        for i in range(len(args)):
            if not isinstance(args[i], Tensor):
                args[i] = Tensor(args[i])
                
        inputs = tuple(a.data for a in args)
        result = obj.forward(*inputs, **kwargs)
        obj.output = result
        for a in args:
            if a.requires_grad == True:
                req_grad = True
                break
            else:
                req_grad = False
        return Tensor(result, parents=args, creator=obj, req_grad=req_grad)

class Mul(operations):
    def forward(self, x, y):
        self.x,self.y = x,y
        return self.x*self.y
    def backward(self, gradient):
        return tuple([self.y*gradient, self.x*gradient])