from tensors import Tensor
import numpy as np
from operations import ReLU

class Linear:
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(in_features, out_features)*np.sqrt(2 / in_features), req_grad=True)
        self.b = Tensor(np.zeros(out_features), req_grad=True)
    def forward(self,x):
        self.result = x @ self.W + self.b
        return self.result
    def parameters(self):
        return (self.W,self.b)
    def __call__(self, x):
        return self.forward(x)


class MLP:
    def __init__(self, input_features, hidden_layer1, hidden_layer2, output_features):
        self.layer_1 = Linear(input_features, hidden_layer1)
        self.layer_2 = Linear(hidden_layer1, hidden_layer2)
        self.layer_3 = Linear(hidden_layer2, output_features)
    def forward(self, x):
        layer1_out = ReLU.run(self.layer_1(x))
        layer2_out = ReLU.run(self.layer_2(layer1_out))
        output = self.layer_3(layer2_out)
        return output
    def __call__(self, x):
        return self.forward(x)
    def parameters(self):
        p1,p2,p3 = self.layer_1.parameters(), self.layer_2.parameters(), self.layer_3.parameters()
        return (p1+p2+p3)

    