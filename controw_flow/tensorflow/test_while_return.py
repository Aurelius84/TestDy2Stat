from tkinter import Y
import numpy as np
import tensorflow as tf

import unittest

class TestIFEarlyReturn(unittest.TestCase):
    def setUp(self):
        self.x = tf.ones([1])
        self.y = tf.ones([1]) * 2

    def test_tf_if(self):
        """
        File "test_while_return.py", line 35, in func  *
            while i < 10:

        NotImplementedError: a return statement cannot be placed inside this TensorFlow loop; this may happen if a return statement depends on a static Python condition such as a hyperparameter
        """

        @tf.function
        def func(x):
            i = tf.constant(1)
            while i < 10:
                i += 1
                if i > 5:
                    x += 110
                    return x
                x += i
            return x
        
        print(tf.autograph.to_code(func.python_function))
        outs = func(self.x)
        print(outs)

        """
        def tf__func(x):
            with ag__.FunctionScope('func', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                i = ag__.converted_call(ag__.ld(tf).constant, (1,), None, fscope)

                def get_state_1():
                    return (do_return, retval_, x, i)

                def set_state_1(vars_):
                    nonlocal do_return, retval_, i, x
                    (do_return, retval_, x, i) = vars_

                def loop_body():
                    nonlocal do_return, retval_, i, x
                    i = ag__.ld(i)
                    i += 1

                    def get_state():
                        return (do_return, retval_, x)

                    def set_state(vars_):
                        nonlocal do_return, retval_, x
                        (do_return, retval_, x) = vars_

                    def if_body():
                        nonlocal do_return, retval_, x
                        x = ag__.ld(x)
                        x += 110
                        try:
                            do_return = True
                            retval_ = ag__.ld(x)
                        except:
                            do_return = False
                            raise

                    def else_body():
                        nonlocal do_return, retval_, x
                        x = ag__.ld(x)
                        x += i
                    ag__.if_stmt((ag__.ld(i) > 5), if_body, else_body, get_state, set_state, ('do_return', 'retval_', 'x'), 3)

                def loop_test():
                    return ag__.and_((lambda : ag__.not_(do_return)), (lambda : (ag__.ld(i) < 10)))
                ag__.while_stmt(loop_test, loop_body, get_state_1, set_state_1, ('do_return', 'retval_', 'x', 'i'), {})

                def get_state_2():
                    return (do_return, retval_)

                def set_state_2(vars_):
                    nonlocal do_return, retval_
                    (do_return, retval_) = vars_

                def if_body_1():
                    nonlocal do_return, retval_
                    try:
                        do_return = True
                        retval_ = ag__.ld(x)
                    except:
                        do_return = False
                        raise

                def else_body_1():
                    nonlocal do_return, retval_
                    pass
                ag__.if_stmt(ag__.not_(do_return), if_body_1, else_body_1, get_state_2, set_state_2, ('do_return', 'retval_'), 2)
                return fscope.ret(retval_, do_return)
        """
        
    
if __name__ == "__main__":
    unittest.main()