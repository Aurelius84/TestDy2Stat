import unittest
import numpy as np
import tensorflow as tf

# 导入tensorflow中autograph的核心module组件
from tensorflow.python.autograph.impl.api import PyToTF
ag__ = PyToTF().get_extra_locals()['ag__']

class TestBlocklocalScope(unittest.TestCase):
    def setUp(self):
        self.x = tf.ones([2,4])
        self.y = tf.ones([2,4]) * 2

    def test_scope(self):

        @tf.function
        def func(x, y, flag=True):
            bs = x.shape[0]
            import pdb;pdb.set_trace()
            if flag:
                # out only visible in if branch
                out = x + y
                # update outer block x
                x = out * 2
            else:
                # update outer block y
                y = x + bs
            # out is a new variable
            out = x + y
            return out

        # 函数签名标记
        out_true = func(self.x, self.y)
        out_false = func(self.x, self.y, False)    
        print(func.pretty_printed_concrete_signatures())
        """
        func(x, y, flag=True)
            Args:
                x: float32 Tensor, shape=(2, 4)
                y: float32 Tensor, shape=(2, 4)
            Returns:
                float32 Tensor, shape=(2, 4)

            func(x, y, flag=False)
            Args:
                x: float32 Tensor, shape=(2, 4)
                y: float32 Tensor, shape=(2, 4)
            Returns:
                float32 Tensor, shape=(2, 4)
        """

        # 转写代码
        print(tf.autograph.to_code(func.python_function))
    

    def pdb_func(self):
        # 可以搭配 import pdb;pdb.set_trace() 来调试
        # 执行 python test_if_basic.py TestBlocklocalScope.pdb_func
        def tf__func(x, y, flag=None):
            with ag__.FunctionScope('func', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                bs = ag__.ld(x).shape[0]            # Load variable operator
                import pdb;pdb.set_trace()

                def get_state():
                    return (x, y)

                def set_state(vars_):
                    nonlocal y, x
                    (x, y) = vars_

                def if_body():
                    nonlocal y, x
                    out = (ag__.ld(x) + ag__.ld(y))
                    x = (ag__.ld(out) * 2)

                def else_body():
                    nonlocal y, x
                    y = (ag__.ld(x) + ag__.ld(bs))
                out = ag__.Undefined('out')         # Represents an undefined symbol in Python
                ag__.if_stmt(ag__.ld(flag), if_body, else_body, get_state, set_state, ('x', 'y'), 2)
                out = (ag__.ld(x) + ag__.ld(y))
                try:
                    do_return = True
                    retval_ = ag__.ld(out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        
        tf__func(self.x, self.y, False)

    
if __name__ == "__main__":
    unittest.main()