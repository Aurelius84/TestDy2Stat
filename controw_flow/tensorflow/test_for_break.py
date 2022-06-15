import numpy as np
import tensorflow as tf

import unittest

class TestForBreak(unittest.TestCase):
    def setUp(self):
        self.x = tf.ones([1])
        self.y = tf.ones([1]) * 2

    def test_break(self):

        @tf.function
        def func(x):
            a = tf.constant(0)
            for i in range(4):  # 需要替换为tf.range
                if a <= 2:   # 不支持python for中包含控制流的break/continue
                    a = a + 1
                    continue
                else:
                    x += 10
                    break
                x = x + 1
            return a
        
        out = func(self.x)
        print(out)
        """
        File "test_for_break.py", line 16, in func  *
            for i in range(4):

        NotImplementedError: break and return statements which depend on a TF condition are not supported in Python for loops. Did you intend to make it a TF loop?
        See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#consistency-of-control-flow-types for more info.
        """
       
        print(tf.autograph.to_code(func.python_function))
        """
        def tf__func(x):
            with ag__.FunctionScope('func', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                a = ag__.converted_call(ag__.ld(tf).constant, (0,), None, fscope)
                break_ = False

                def get_state_2():
                    return (a, break_, x)

                def set_state_2(vars_):
                    nonlocal a, x, break_
                    (a, break_, x) = vars_

                def loop_body(itr):
                    nonlocal a, x, break_
                    i = itr
                    continue_ = False
                    (break_,)

                    def get_state():
                        return (a, break_, continue_, x)

                    def set_state(vars_):
                        nonlocal a, continue_, x, break_
                        (a, break_, continue_, x) = vars_

                    def if_body():
                        nonlocal a, continue_, x, break_
                        a = (ag__.ld(a) + 1)
                        continue_ = True

                    def else_body():
                        nonlocal a, continue_, x, break_
                        x = ag__.ld(x)
                        x += 10
                        break_ = True
                        continue_ = True
                    ag__.if_stmt((ag__.ld(a) <= 2), if_body, else_body, get_state, set_state, ('a', 'break_', 'continue_', 'x'), 4)

                    def get_state_1():
                        return (x,)

                    def set_state_1(vars_):
                        nonlocal x
                        (x,) = vars_

                    def if_body_1():
                        nonlocal x
                        x = (ag__.ld(x) + ag__.ld(i))

                    def else_body_1():
                        nonlocal x
                        pass
                    ag__.if_stmt(ag__.not_(continue_), if_body_1, else_body_1, get_state_1, set_state_1, ('x',), 1)

                def extra_test():
                    nonlocal a, x, break_
                    return ag__.not_(break_)
                i = ag__.Undefined('i')
                continue_ = ag__.Undefined('continue_')
                ag__.for_stmt(ag__.converted_call(ag__.ld(range), (4,), None, fscope), extra_test, loop_body, get_state_2, set_state_2, ('a', 'break_', 'x'), {'iterate_names': 'i'})
                try:
                    do_return = True
                    retval_ = ag__.ld(a)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        """
        

class TestNestFor(unittest.TestCase):
    def setUp(self):
        self.x = tf.ones([1])
        self.y = tf.ones([1]) * 2

    def test_break(self):

        @tf.function
        def func(x):
            s = 0
            for i in range(4):  # 需要替换为tf.range
                x  = x - 1
                for i in range(4):
                    x = x - 10
            return s
        
        out = func(self.x)
        print(out)
       
        print(tf.autograph.to_code(func.python_function))
    
if __name__ == "__main__":
    unittest.main()