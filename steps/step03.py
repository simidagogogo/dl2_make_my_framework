from step02 import Variable, Function, Square
import numpy as np

class Exp(Function):
    def forward(self, x: np.ndarray):
        return np.exp(x)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

print(y.data)