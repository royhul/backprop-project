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

def grad_check(ops, *args):
    obj = ops()
    args = list(args)
    for i in range(len(args)):
        if not isinstance(args[i], Tensor):
            args[i] = Tensor(args[i])
    eps = Tensor(1e-6)
    # args = tuple(a.data for a in args)
    def nudged(arr,idx,delta):
        out = arr.copy()
        out[idx] += delta
        return out
    for i in range(len(args)):
        for idx in np.ndindex(args[i].data.shape):
            inputs = tuple(nudged(args[j].data, idx, eps.data) if j == i else args[j].data for j in range(len(args)))
            psv_res = obj.forward(*inputs)
            inputs = tuple(nudged(args[j].data, idx, -eps.data) if j == i else args[j].data for j in range(len(args)))
            neg_res = obj.forward(*inputs)
            grad = np.sum((psv_res - neg_res)/(2*eps.data))
            rel_error = abs(grad - args[i].grad[idx])/np.maximum(np.maximum(abs(grad),abs(args[i].grad[idx])),1e-12)
            if np.max(rel_error) >= 1e-5:
                return False
    return True

class Mul(operations):
    def forward(self, x, y):
        self.x,self.y = x,y
        return self.x*self.y
    def backward(self, gradient):
        return tuple([self.y*gradient, self.x*gradient])