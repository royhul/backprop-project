import numpy as np

class Tensor:

    def __init__(self, data, parents=(), creator=None, req_grad=True):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.parents = parents
        self.creator = creator
        self.requires_grad = req_grad

    def __repr__(self):
        op_name = "empty" if self.creator is None else f"{type(self.creator).__name__}"
        return f"Data={self.data}, grad={self.grad}, op={op_name}, requires_grad={self.requires_grad}"
    
    @property
    def shape(self):
        return self.data.shape
    
    def backward(self):
        topo_arr = []
        self.topo_sort(set(),set(), topo_arr)
        topo_arr.reverse()
        for i in range(len(topo_arr)):
            if topo_arr[i].creator is None:
                continue
            else:
                topo_arr[i].grad = np.zeros_like(topo_arr[i].data)
        self.grad = np.ones_like(self.data)
        for i in range(len(topo_arr)):
            if topo_arr[i].creator is None:
                continue
            parent_gradients = topo_arr[i].creator.backward(topo_arr[i].grad)
            for n,s in zip(topo_arr[i].parents, parent_gradients):
                if n.grad.shape != s.shape:
                    s = self.unbroadcast(n,s)
                n.grad += s

    def unbroadcast(self, tgt_grad, act_grad):
        for i in range(len(act_grad.shape) - len(tgt_grad.shape)):
            act_grad = np.sum(act_grad, axis = 0)
        if len(tgt_grad.shape) != 0:
            for j in range(len(tgt_grad.shape)):
                if act_grad.shape[j] != tgt_grad.shape[j]:
                    act_grad = np.sum(act_grad, axis = j, keepdims=True)

        return act_grad    

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def topo_sort(self, visiting, visited, result):
        visiting.add(self)
        for i in range(len(self.parents)):
            if self.parents[i] not in visited:
                self.parents[i].topo_sort(visiting, visited, result)
        visiting.remove(self)
        visited.add(self)
        result.append(self)
        return
    