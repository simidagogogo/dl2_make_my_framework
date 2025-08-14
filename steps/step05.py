from step02 import Variable, Function
from step03 import Square, Exp
import numpy as np

# 链式法则（链式规则）
# 在前面的步骤中，我们使用数值微分计算了复合函数的导数
# 现在让我们了解如何使用链式法则手动计算导数

# 复合函数 y = (e^(x^2))^2 的导数
# 设 a = x^2, b = e^a, y = b^2
# 根据链式法则: dy/dx = (dy/db) * (db/da) * (da/dx)

# 各个函数的导数:
# da/dx = 2x (x^2的导数)
# db/da = e^a (e^x的导数)  
# dy/db = 2b (x^2的导数)

# 因此: dy/dx = 2b * e^a * 2x = 2 * e^(x^2) * e^(x^2) * 2x = 4x * e^(2*x^2)

class Variable:
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data

def f(x: Variable):
    """复合函数: y = (e^(x^2))^2"""
    A = Square()  # x^2
    B = Exp()     # e^x
    C = Square()  # x^2
    return C(B(A(x)))

def manual_backward():
    """手动计算反向传播 - 演示链式法则"""
    print("=== 手动反向传播示例 ===")
    
    # 正向传播
    x = Variable(np.array(0.5))
    print(f"输入 x = {x.data}")
    
    # 分步计算，保存中间结果
    A = Square()
    B = Exp() 
    C = Square()
    
    a = A(x)  # a = x^2
    print(f"a = x^2 = {a.data}")
    
    b = B(a)  # b = e^a
    print(f"b = e^a = {b.data}")
    
    y = C(b)  # y = b^2  
    print(f"y = b^2 = {y.data}")
    
    print("\n=== 反向传播（链式法则）===")
    
    # 反向传播 - 手动计算梯度
    # 从输出开始，逐步向输入传播
    
    # dy/dy = 1 (输出对自身的导数)
    gy = 1.0
    print(f"dy/dy = {gy}")
    
    # dy/db = 2 * b (因为 y = b^2)
    gb = 2 * b.data * gy
    print(f"dy/db = 2 * b * dy/dy = 2 * {b.data} * {gy} = {gb}")
    
    # db/da = e^a (因为 b = e^a)  
    ga = np.exp(a.data) * gb
    print(f"db/da = e^a * dy/db = e^{a.data} * {gb} = {ga}")
    
    # da/dx = 2 * x (因为 a = x^2)
    gx = 2 * x.data * ga  
    print(f"da/dx = 2 * x * db/da = 2 * {x.data} * {ga} = {gx}")
    
    print(f"\n最终结果: dy/dx = {gx}")
    return gx

def compare_with_numerical():
    """与数值微分结果比较"""
    print("\n=== 与数值微分比较 ===")
    
    # 使用step04的数值微分
    def numerical_diff(f, x, eps=1e-4):
        x0 = Variable(x.data - eps)
        x1 = Variable(x.data + eps) 
        y0 = f(x0)
        y1 = f(x1)
        return (y1.data - y0.data) / (2 * eps)
    
    x = Variable(np.array(0.5))
    
    # 手动计算的结果
    manual_result = manual_backward()
    
    # 数值微分的结果
    numerical_result = numerical_diff(f, x)
    
    print(f"手动链式法则结果: {manual_result}")
    print(f"数值微分结果: {numerical_result}")
    print(f"误差: {abs(manual_result - numerical_result)}")

if __name__ == "__main__":
    print("Step 05: 链式法则和手动反向传播")
    print("=" * 50)
    
    # 演示链式法则的手动计算
    manual_backward()
    
    # 与数值微分比较
    compare_with_numerical()
    
    print("\n" + "=" * 50)
    print("下一步：step06 将实现自动反向传播")