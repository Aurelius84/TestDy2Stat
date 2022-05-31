from tkinter import Y
import numpy as np
import tensorflow as tf

import unittest

# LEGB规则: 在本地空间寻找不到的变量，逐级向上级寻找。LEGB 分别指代Local，Enclose，Global 和 Builtin
class TestPythonNonLocal(unittest.TestCase):
    def setUp(self):
        self.x = tf.ones([2,4])
        self.y = tf.ones([2,4]) * 2
    

    def test_non_local(self):
        """
        test for nonlocal simple mechanism.
        """

        def func(x, y):
            c = x + y

            def add():
                c = x - y  # local var, not same with func.c
            add()
            return c
        
        def func_nonlocal(x, y):
            c = x + y

            def add_to():
                nonlocal c
                c = x - y  # free var, same with func.c
            
            add_to()
            return c
        
        out1 = func(1, 2)
        out2 = func_nonlocal(1, 2)

        self.assertEqual(out1, 3)
        self.assertEqual(out2, -1)


    def test_cfg_if_non_local(self):
        """
        Simulate control flow IF with nonlocal mechanism.
        
        y = 0
        if x > 10:
            y = x + 1
        else:
            y = x - 1
        """

        def if_api(cond, body, orelse):
            return body() if cond else orelse()
        
        def simple_if(x):

            def cond():
                nonlocal x
                return x > 10
            
            def true_fn():
                nonlocal x, y
                y = x + 1
                print("true_fn: id(x), id(y): ", id(x), id(y))
                return y
            
            def false_fn():
                nonlocal x, y
                y = x - 1
                print("false_fn: id(x), id(y): ", id(x), id(y))
                return y
            
            y = 0   # we should firstly create y to make `nonlocal y` valid.
            out = if_api(cond(), true_fn, false_fn)
            print("outer_fn: id(x), id(y): ", id(x), id(y))
            return out
        
        out1 = simple_if(12) # y = x + 1
        out2 = simple_if(5)  # y = x - 1
        self.assertEqual(out1, 12 + 1)
        self.assertEqual(out2, 5 - 1)


class TestNonLocalIF(unittest.TestCase):
    def setUp(self):
        self.x = tf.ones([1])
        self.y = tf.ones([1]) * 2

    def test_nonlocal(self):

        @tf.function
        def func(x, y):
            avg_x = tf.reduce_mean(x)
            if avg_x > 10:
                # out only visible in if branch
                out = y + 1
                # update outer block x
                x = out * 2
            else:
                # update outer block y
                out = x + 1
                y = out * 2

            return x, y, out
        
        x, y, out = func(self.x, self.y)
        self.assertEqual(x[0], 1)
        self.assertEqual(y[0], 4)
        self.assertEqual(out[0], 2)

        print(tf.autograph.to_code(func.python_function))
        """
        def tf__func(x, y):
            with ag__.FunctionScope('func', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                avg_x = ag__.converted_call(ag__.ld(tf).reduce_mean, (ag__.ld(x),), None, fscope)

                def get_state():
                    return (out, x, y)

                def set_state(vars_):
                    nonlocal y, x, out
                    (out, x, y) = vars_

                def if_body():
                    nonlocal y, x, out
                    out = (ag__.ld(y) + 1)
                    x = (ag__.ld(out) * 2)

                def else_body():
                    nonlocal y, x, out
                    out = (ag__.ld(x) + 1)
                    y = (ag__.ld(out) * 2)
                out = ag__.Undefined('out')
                ag__.if_stmt((ag__.ld(avg_x) > 10), if_body, else_body, get_state, set_state, ('out', 'x', 'y'), 3)
                try:
                    do_return = True
                    retval_ = (ag__.ld(x), ag__.ld(y), ag__.ld(out))
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        """
        
    
if __name__ == "__main__":
    unittest.main()