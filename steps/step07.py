# 接下来要做的就是让反向传播自动化
# 准确来说，就是要建立这样一个机制:m无论普通的计算流程(正向传播)中是什么样的计算，反向传播都能自动进行
# 我们马上要接触到 Define-by-Run的核心了
# Define-by-Run是在深度学习中进行计算时，在计算之间建立"连接"的机制, 这种机制也称为动态计算图

# 7-1所示的计算图都是流水线式的计算, 因此，只要以列表的形式记录函数的顺序，就可以通过反向回溯自动进行反向传播
# 不过，对于有分支的计算图或多次使用同一个变量的复杂计算图，只借助简单的列表就不能奏效了
# 我们接下来的目标是建立一个不管计算图多么复杂，都能自动进行反向传播的机制

# 为反向传播的自动化创造条件
# 在实现反向传播的自动化之前，我们先思考一下变量和函数之间的关系 。 首先从函数的角度来考虑， 即思考"从函数的角度如何看待变量"。 从函数 的角度来看，变量是以输入和输出的形式存在的 。 如 同 Î-'2左图所示，函数 的变量包括"输入变量 "(i叩ut)和"输出变量"( output)(图中的虚线表示引用)。
# 刀15么从变i茸的角度来看 ， 雨数是什么样的呢? 这里要强调的是变址是 81 函数"创造 " 的 。 也就是说，雨数是变茧的"父母"，是 creator(创造者)。 如 果变量没有作为创造者的函数，我们就可以认为它是由非函数创造的，比如用户给出的变量 。

# 下面在代码中实现7-2所示的函数和变量之间的"连接" 
# 我们让这个"连接"在执行普通计算(正向传播)的那一刻创建
# 为此, 先在 Variable类巾添加以下代码

import numpy as np

class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func: 'Function'):
        self.creator = func

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
    
# DeZero的动态计算图的原理是在执行实际的计算时，在变量这个“箱子"里记录它的“连接”
# Chainer和PyTorch也采用了类似的机制
# 这样一来, Variable 和 Function之间就有了"连接", 我们就可以反向遍历计算图了

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

def run1():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # 反向遍历计算图的节点
    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator ==  B
    assert y.creator.input.creator.input ==  a
    assert y.creator.input.creator.input.creator ==  A
    assert y.creator.input.creator.input.creator.input ==  x

    # 计算图是由函数和变量之间的"连接"构建而成
    # 更重要的是，这个"连接"是在计算实际发生的时候(数据在正向传播流转的时候)形成的, 变量和函数连接的这个特征就是 Define-by-Run
    # 换言之，"连接" 是通过数据的流转建立起来的

    y.grad = np.array(1.0)
    C = y.creator               # 1.获取函数
    b = C.input                 # 2.获取函数的输入
    b.grad = C.backward(y.grad) # 3.调用函数的backward方法

    B = b.creator
    a = B.input
    a.grad = B.backward(b.grad)

    # 上述代码执行的反向传播的逻辑与之前的相同
    # 具体来说，该流程如下
    # 1.获取函数
    # 2.获取函数的输入
    # 3.调用函数的 backward方法

    A = a.creator               # 1.获取函数
    x = A.input                 # 2.获取函数的输入
    x.grad = A.backward(a.grad) # 3.调用函数的backward方法
    print("run1: ", x.grad)


# 从前面这些反向传播的代码可以看出，它们有着相同的处理流程 
# 准确来说，是从一个变量到前一个变量反向传播逻辑相同 
# 为了完成这些重复的处理，我们在 Variable类中添加一个新的方法一- backward
class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func: Function):
        self.creator = func

    def backward(self):
        """
        backward方法和此前反复出现的流程基本相同 
        具体米说，它从 Variable的creator获取函数并取出函数的输入变量, 然后调用函数的backward方法, 最后,它会针对自己前面的变量, 调用它的 backward方法, 这样每个变量的 backward方法就会被递归调用
        如果Variable实例的creator是None，那么反向传播就此结束。这种情况意味着Variable实例是由非函数创造的，主要来自用户提供的变量。
        """
        f = self.creator                # 1.获取函数
        if f is None:
            return
        x = f.input                     # 2.获取函数的输入
        x.grad = f.backward(self.grad)  # 3.调用函数的backward方法
        x.backward()                    # 调用自己前面那个变量的backward方法(递归)

def run2():
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
    print("run2: ", x.grad)

if __name__ == "__main__":
    run1()
    run2()

# 只要像上面那样调用变量y的backward方法，反向传播就会自动进行 。 运行结果和之前的一样 。 这样我们 就打好了对 DeZero来说最重要的自动微分的基础
