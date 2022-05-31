from tkinter import Y
import numpy as np
import tensorflow as tf

import unittest

class TestIFEarlyReturn(unittest.TestCase):
    def setUp(self):
        self.x = tf.ones([1])
        self.y = tf.ones([1]) * 2

    def test_nonlocal(self):

        @tf.function
        def func(x, y):
            avg_x = tf.reduce_mean(x)
            if avg_x > 10:
                out = y + 1
                x = out * 2
                return x, out
            else:
                # update outer block y
                out = x + 1
                y = out * 2

            return y, out
        
        x, y = func(self.x, self.y)
        self.assertEqual(x[0], 4)
        self.assertEqual(y[0], 2)
       
        print(tf.autograph.to_code(func.python_function))
        """
        def tf__func(x, y):
            with ag__.FunctionScope('func', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                avg_x = ag__.converted_call(ag__.ld(tf).reduce_mean, (ag__.ld(x),), None, fscope)

                def get_state():
                    return (do_return, retval_, x, y)

                def set_state(vars_):
                    nonlocal x, retval_, y, do_return
                    (do_return, retval_, x, y) = vars_

                def if_body():
                    nonlocal x, retval_, y, do_return
                    out = (ag__.ld(y) + 1)
                    x = (ag__.ld(out) * 2)
                    try:
                        do_return = True
                        retval_ = (ag__.ld(x), ag__.ld(out))
                    except:
                        do_return = False
                        raise

                def else_body():
                    nonlocal x, retval_, y, do_return
                    out = (ag__.ld(x) + 1)
                    y = (ag__.ld(out) * 2)
                    try:
                        do_return = True
                        retval_ = (ag__.ld(y), ag__.ld(out))
                    except:
                        do_return = False
                        raise
                out = ag__.Undefined('out')             # out need to be defined outside

                # No try-except here and removed into if-body and else_body
                ag__.if_stmt((ag__.ld(avg_x) > 10), if_body, else_body, get_state, set_state, ('do_return', 'retval_', 'x', 'y'), 2)
                return fscope.ret(retval_, do_return)
        """
        
    
if __name__ == "__main__":
    unittest.main()