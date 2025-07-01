import numpy as np

class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data

class Function:
    def __call__(self, input: Variable):
        """
        call 方法执行两项任务 : 从 Variable取出数据和将计算结果保存到Variable中
        其中具体的计算是 通过forward方法完成
        """
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        """
        forward方法的实现会在继承类中完成
        """
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
if __name__ == "__main__":
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)