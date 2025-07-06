
# DeZero现在叮以通过反向传播进行计算了 。 它j圣拥有一项名为 Define­ by-Run的能力，可以在运行时在计算之间建立"连接"。 为了使 DeZero更加 劫用 、 本步骤将对 DeZero的函数进行 3项改进 

# 此前, DeZero中使用的函数是作为 Python的类实现的 。 举例来说 ，在 使用 Square类进行计算的情况下，我们需要编写如下代码
from step08 import Variable, Function, Square, Exp
import numpy as np

x = Variable(np.array(0.5))
f = Square()
y = f(x)

# 上面的代码是分两步计算平方的：创建一个Square类的实例；调用这个实例。
# 但是从用户的角度来看，分两步完成有点啰唆(虽然可以写成y=Square()(x)，但观感很差)
# 用户更希望把DeZero的函数当作Python函数使用。为此，需要添加以下代码

