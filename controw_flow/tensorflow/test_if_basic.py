import numpy as np
import tensorflow as tf

import unittest

class TestBlocklocalScope(unittest.TestCase):
    def setUp(self):
        # self.x = tf.ones([2,4])
        # self.y = tf.ones([2,4]) * 2
        pass

    def test_scope(self):
        print(1)
        # @tf.function
        # def func(x, y, flag=True):
        #     bs = x.shape[0]
        #     if flag:
        #         # out only visible in if branch
        #         out = x + y
        #         # update outer block x
        #         x = out * 2
        #     else:
        #         # update outer block y
        #         y = x + bs
        #     # out is a new variable
        #     out = x + y
        #     return out
        
        # out = func(self.x, self.y)
        # print(out)
        
    
if __name__ == "__main__":
    unittest.main()