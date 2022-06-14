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
            max_len = tf.constant(3)
            for i in range(max_len):
                if i == 2:
                    break
                x = x + i
            return x
        
        # out = func(self.x)
       
        print(tf.autograph.to_code(func.python_function))
        """
        def tf__func(x):
            with ag__.FunctionScope('func', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                max_len = ag__.converted_call(ag__.ld(tf).constant, (3,), None, fscope)
                break_ = False

                def get_state_2():
                    return (x, break_)

                def set_state_2(vars_):
                    nonlocal x, break_
                    (x, break_) = vars_

                def loop_body(itr):
                    nonlocal x, break_
                    i = itr
                    continue_ = False
                    (break_,)

                    def get_state():
                        return (break_, continue_)

                    def set_state(vars_):
                        nonlocal continue_, break_
                        (break_, continue_) = vars_

                    def if_body():
                        nonlocal continue_, break_
                        break_ = True
                        continue_ = True

                    def else_body():
                        nonlocal continue_, break_
                        pass
                    ag__.if_stmt((ag__.ld(i) == 2), if_body, else_body, get_state, set_state, ('break_', 'continue_'), 2)

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
                    nonlocal x, break_
                    return ag__.not_(break_)
                continue_ = ag__.Undefined('continue_')
                i = ag__.Undefined('i')
                ag__.for_stmt(ag__.converted_call(ag__.ld(range), (ag__.ld(max_len),), None, fscope), extra_test, loop_body, get_state_2, set_state_2, ('x', 'break_'), {'iterate_names': 'i'})
                try:
                    do_return = True
                    retval_ = ag__.ld(x)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        """
        
    
if __name__ == "__main__":
    unittest.main()