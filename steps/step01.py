import numpy as np

class Variable: 
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data

data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)
print(f"ndim:", x.data.ndim)

class Function:
    # call方法是一个特殊的Python方法
    # 定义这个方法后，当f=Function()时，可通过编写f(...)调用__call__方法
    def __call__(self, input: Variable) -> Variable:
        data = input.data
        y = data ** 2
        output = Variable(y)
        return output

x = Variable(np.array(10)) 
f = Function()
y = f(x)
print(type(y))
print(y.data)
