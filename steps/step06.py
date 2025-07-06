import numpy as np

# 数值微分在计算戚本和精度方面存在问题
# 反向传播可以解决这两个问题. 也就是说, 反向传播不仅能 高效地求导, 还能帮助我们得到误差更小的值

# 理解反向传播的关键是链式法则（连锁律）
# 链（chain）可以理解为链条、锁链等，在这里表示多个函数连接在一起使用
# 链式法则意为连接起来的多个函数（复合函数）的导数可以分解为各组成函数的导数的乘积

# y到x的导数可以表示为各函数的导数的乘积. 
# 换言之, 复合函数的导数可以分解为各组成的数导数的乘积, 这就是链式法则

# 下面实现支持反向传播的 Variable类。
# 为此，我们要扩展Variable类，除普通值 (data)之外，增加与之对应的导数值grad

class Variable:
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data
        self.grad = None
    
class Function:
    def __call__(self, input: Variable) -> Variable:
        self.input = input # 保存输入的变量
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x: np.ndarray):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x ** 2
        return y

    def backward(self, gy) -> np.ndarray:
        # backward返问的结果是通过这个参数传播来的导数和"y=x^2的导数"的乘积
        # 这个返回结果会进一步向输入方向传播
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
        
if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    # 正向传播的代码
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # 反向传播计算y的导数
    # 按照与正向传播相反的顺序调用各函数的 backward方法 
    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)
    