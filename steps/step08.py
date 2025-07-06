# 在上一个步骤中. 我们向Variable类添加了backward方法
# 考虑到处理效率的改善和今后的功能扩展，本步骤将改进 backward方法的实现方式

import numpy as np

class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func: 'Function'):
        self.creator = func

    # def backward(self):
    #     """
    #     backward方法和此前反复出现的流程基本相同 
    #     具体米说，它从 Variable的creator获取函数并取出函数的输入变量, 然后调用函数的backward方法, 最后,它会针对自己前面的变量, 调用它的 backward方法, 这样每个变量的 backward方法就会被递归调用
    #     如果Variable实例的creator是None，那么反向传播就此结束。这种情况意味着Variable实例是由非函数创造的，主要来自用户提供的变量。
    #     """
    #     f = self.creator                # 1.获取函数
    #     if f is None:
    #         return
    #     x = f.input                     # 2.获取函数的输入
    #     x.grad = f.backward(self.grad)  # 3.调用函数的backward方法
    #     x.backward()                    # 调用自己前面那个变量的backward方法(递归)

    def backward(self):
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

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # 反向传播
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

# 和上一个步骤得到的结果一样。这样实现方式就从递归变成了循环。在步骤15，我们将感受到循环带来的好处。届时要处理的是复杂的计算图，不过在使用循环的情况下，代码实现很容易扩展到复杂的计算图的处理，而且循环的执行效率也会变高。
# 每次递归调用函数时，函数都会将处理过程中的结果保留在内存中（或者说保笛在栈中），然后迷续处理。因此一般来说循环的效率更高. 对现代计算机来说，使用少量的内存是没有问题的。有时可以通过尾递归的技巧，使递归处理能够按照循环的方式执行。