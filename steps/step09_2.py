import numpy as np

# 9.2 简化backward方法
# 第2项要改进的地方是减少用户在反向传播方面所做的工作
# 具体来说, 就是省略前面代码中的y.grad = np.array(1.0)
# 每次反向传播时我们都要重新编写这行代码

class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func: 'Function'):
        self.creator = func

    def backward(self):
        # 如果变量grad为None, 则自动生成导数.
        # 代码中通过np.ones_like(self.data)创建了一个ndarray实例.
        # 该实例的形状和数据类型与self.data的相同.
        # 如果self.data是标量, 那么self.grad也是标量.
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()             # 获取函数
            x, y = f.input, f.output    # 获取函数的输入
            x.grad = f.backward(y.grad) # backward调用 backward方法
            if x.creator is not None:
                funcs.append(x.creator) # 将前一个函数添加到列表中

class Function:
    def __call__(self, input: Variable) -> Variable:
        self.input: Variable = input
        x = input.data
        y = self.forward(x)

        # 对于创建的 output变量，代码让它保存了"我(函数本身)是创造者"的信息. 这是动态建立"连接"机制的核心
        output = Variable(y)
        output.set_creator(self) # 让输出变量保存创造者信息
        self.output:Variable = output # 也保存输出变量
        return output
    
    def forward(self, x: np.ndarray):
        raise NotImplementedError()
    
    def backward(self, gy: np.ndarray):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x ** 2
        return y

    def backward(self, gy) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x) -> np.ndarray:
        y = np.exp(x)
        return y
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# 9.1 作为Python函数使用
# 目的:  将DeZero的函数当作Python函数使用了
def square(x: Variable):
    f: Function = Square()
    return f(x)

def exp(x: Variable):
    f = Exp()
    return f(x)

def run2():
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)

if __name__ == "__main__":
    run2()