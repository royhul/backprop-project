import numpy as np
import tensors

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
            if not isinstance(args[i], tensors.Tensor):
                args[i] = tensors.Tensor(args[i])
                
        inputs = tuple(a.data for a in args)
        result = obj.forward(*inputs, **kwargs)
        obj.output = result
        for a in args:
            if a.requires_grad == True:
                req_grad = True
                break
            else:
                req_grad = False
        return tensors.Tensor(result, parents=args, creator=obj, req_grad=req_grad)


def grad_check(ops, *args):
    obj = ops()
    args = list(args)
    for i in range(len(args)):
        if not isinstance(args[i], tensors.Tensor):
            args[i] = tensors.Tensor(args[i])
    eps = tensors.Tensort(1e-6)
    def nudged(arr,idx,delta):
        out = arr.copy()
        out[idx] += delta
        return out
    for i in range(len(args)):
        if args[i].requires_grad == False:
            continue
        for idx in np.ndindex(args[i].data.shape):
            inputs = tuple(nudged(args[j].data, idx, eps.data) if j == i else args[j].data for j in range(len(args)))
            psv_res = obj.forward(*inputs)
            inputs = tuple(nudged(args[j].data, idx, -eps.data) if j == i else args[j].data for j in range(len(args)))
            neg_res = obj.forward(*inputs)
            grad = np.sum((psv_res - neg_res)/(2*eps.data))
            rel_error = abs(grad - args[i].grad[idx])/np.maximum(np.maximum(abs(grad),abs(args[i].grad[idx])),1e-12)
            if np.isnan(rel_error).any():
                raise RuntimeError("Gradient is undefined")
            if np.max(rel_error) >= 1e-5:
                return False
    return True

class Mul(operations):
    def forward(self, x, y):
        self.x,self.y = x,y
        return self.x * self.y
    def backward(self, gradient):
        return tuple([self.y*gradient, self.x*gradient])

class Add(operations):
    def forward(self, x, y):
        self.x,self.y = x,y
        return self.x + self.y
    def backward(self, gradient):
        return tuple([gradient, gradient])
    
class Sub(operations):
    def forward(self, x, y):
        self.x,self.y = x,y
        return self.x - self.y
    def backward(self, gradient):
        return tuple([gradient, -gradient])
    
class Div(operations):
    def forward(self, x, y):
        self.x,self.y = x,y
        return self.x * (self.y ** -1)
    def backward(self, gradient):
        return tuple([gradient*(self.y ** -1), gradient*self.x*(-self.y ** -2)])

class Pow(operations):
    def forward(self, x, y):
        self.x,self.y = x,y
        return self.x ** self.y
    def backward(self, gradient):
        return tuple([self.y * (self.x**(self.y - 1)) * gradient, (self.x ** self.y)*np.log(self.x)*gradient])

class Neg(operations):
    def forward(self, x):
        return -x
    def backward(self, gradient):
        return tuple([-gradient])

class Exp(operations):
    def forward(self, x):
        self.x = x
        return np.exp(self.x)
    def backward(self, gradient):
        return tuple([gradient*np.exp(self.x)])

class Log(operations):
    def forward(self, x):
        self.x = x
        return np.log(self.x)
    def backward(self, gradient):
        return tuple([gradient*(self.x**(-1))])

class ReLU(operations):
    def forward(self, x):
        self.x = x
        return np.maximum(self.x, 0)
    def backward(self, gradient):
        mask = self.x > 0
        result = np.where(mask, gradient, 0)
        return tuple([result])

class Tanh(operations):
    def forward(self, x):
        return np.tanh(x)
    def backward(self, gradient):
        return tuple([(1-self.output**2)*gradient])

class Sum(operations):
    def forward(self, x, axis=None, keepdims=False):
        self.x = x
        self.axis = axis
        self.keepdims = keepdims
        return np.sum(self.x, axis=self.axis, keepdims=self.keepdims)
    def backward(self, gradient):
        result = gradient
        if self.axis == None:
            axes = tuple(range(self.x.ndim))
        elif type(self.axis) == int:
            axes = tuple([self.axis % self.x.ndim])
        else:
            axes = []
            for ele in self.axis:
                axes.append(ele % self.x.ndim)
            axes.sort()
            axes = tuple(axes)
        if self.keepdims == False:
            for ele in axes:
                result = np.expand_dims(result, ele)
        result = np.broadcast_to(result, self.x.shape)
        return tuple([result])

class Mean(operations):
    def forward(self, x, axis=None, keepdims=False):
        self.x = x
        self.axis = axis
        self.keepdims = keepdims
        return np.mean(self.x, axis=self.axis, keepdims=self.keepdims)
    def backward(self, gradient):
        result = gradient
        if self.axis == None:
            axes = tuple(range(self.x.ndim))
            result = result/(np.prod([self.x.shape[k] for k in axes]))
        elif type(self.axis) == int:
            axes = tuple([self.axis % self.x.ndim])
            result = result/(np.prod([self.x.shape[k] for k in axes]))
        else:
            axes = []
            for ele in self.axis:
                axes.append(ele % self.x.ndim)
            axes.sort()
            axes = tuple(axes)
            result = result/(np.prod([self.x.shape[k] for k in axes]))
        if self.keepdims == False:
            for ele in axes:
                result = np.expand_dims(result, ele)
        result = np.broadcast_to(result, self.x.shape)
        return tuple([result])

class MatMul(operations):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x @ self.y
    def backward(self, gradient):
        if self.x.ndim == 1:
            a = self.x.reshape(1,-1)
        else:
            a = self.x
        if self.y.ndim == 1:
            b = self.y.reshape(-1,1)
        else:
            b = self.y
        target = (a.shape[0],b.shape[1])
        if gradient.shape != target:
            gradient = gradient.reshape(target)
        grad_a,grad_b =  gradient@b.T, a.T@gradient
        grad_a,grad_b = grad_a.reshape(self.x.shape),grad_b.reshape(self.y.shape)                                                                                        
        return tuple([grad_a,grad_b])
    
class Reshape(operations):
    def forward(self, x, *, shape):
        self.x = x
        result = self.x.reshape(shape)
        return result
    def backward(self, gradient):
        result = gradient.reshape(self.x.shape)
        return tuple([result])

class Transpose(operations):
    def forward(self, x, axes = None):
        self.x = x
        self.axes = []
        if axes != None:
            for ele in axes:
                if ele < 0:
                    self.axes.append(ele % self.x.ndim)
                else:
                    self.axes.append(ele)
        else:
            self.axes = None
        return self.x.transpose(self.axes)
    def backward(self, gradient):
        if self.axes != None:
            inv = np.argsort(self.axes)
        else:
            inv = None
        return tuple([gradient.transpose(inv)])

class SoftmaxCrossEntropy(operations):
    def forward(self, logits,*, labels):
        #Softmax
        self.logits = logits - np.max(logits, axis=1, keepdims=True) 
        self.den = np.sum(np.exp(self.logits), axis=1, keepdims=True)
        #CrossEntropy
        self.labels = np.eye(self.logits.shape[1])[labels]
        log_prob = self.logits-np.log(self.den)
        cross_ent = np.mean(np.sum(-self.labels*log_prob, axis=1))
        return cross_ent
    def backward(self, gradient):
        axes = tuple(range(self.logits.ndim))
        gradient = gradient/(self.logits.shape[0])
        log_prob = self.logits-np.log(self.den)
        return tuple([(np.exp(log_prob)-self.labels)*gradient])
        