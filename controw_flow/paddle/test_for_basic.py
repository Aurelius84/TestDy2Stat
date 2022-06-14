import numpy as np
import paddle
from paddle.jit import to_static

import unittest

class TestBlocklocalScope(unittest.TestCase):
    def setUp(self):
        self.x = paddle.to_tensor([3, 2, 4])
        self.y = paddle.to_tensor([1, 3, 2, 3])
        pass
    
    """
    Under this case, type hint of 'flag' is needed,
    otherwise it will raise error. Because Torch will
    infer type(flag) as Tensor. The error msg as follows:

    RuntimeError: 
        Expected a default value of type Tensor (inferred) on parameter "flag".Because "flag" was not annotated with an explicit type it is assumed to be type 'Tensor'.:
    """
    def test_for_range_start(self):
        @to_static
        def func(x, y):
            """ 
            PASS: 因为 paddle 中的 ForNodeVisitor 会进行转换，如果range有两个参数那么会选择前面的作为初始值，后面的作为cond值。
            """
            s = 0
            for i in range(10,20):
                s += 1
            return s
        #print(func.code)
        self.assertEqual(10, func(self.x, self.y))

    def test_for_range_same_name(self):
        """ FAILED
        """
        @to_static
        def func(x, y):
            """ 
            FAILED: 因为for转为了while，while形式下i会决定迭代，而for的迭代不是由i决定的
            """
            s = 0
            for i in range(4):
                for i in range(4):
                    s += 1
            return s
        #print(func.code)
        self.assertEqual(16, func(self.x, self.y))

    def test_for_range_idx(self):
        """ FAILED
        """
        @to_static
        def func(x, y):
            """ 
            FAILED: 
            """
            for i in range(4):
                pass
            return i
        #print(func.code)
        self.assertEqual(3, func(self.x, self.y))
    
if __name__ == "__main__":
    unittest.main()
