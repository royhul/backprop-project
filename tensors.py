import numpy as np

class Tensor:
    def __init__(self, data , parents=(), creator=None, req_grad=True, dtype=np.float64):
        self.data = np.asarray(data, dtype = dtype)
        self.grad = np.zeros_like(self.data)
        self.parents = parents
        self.creator = creator
        self.requires_grad = req_grad
    def __repr__(self):
        op_name = "empty" if self.creator is None else f"{type(self.creator).__name__}"
        return f"Data={self.data}, grad={self.grad}, op={op_name}, requires_grad={self.requires_grad}"
    def backward(self):
            parent_gradients = self.creator.backward(self.grad)
            for i in range(len(self.parents)):
                self.parents[i].grad = parent_gradients[i]
    