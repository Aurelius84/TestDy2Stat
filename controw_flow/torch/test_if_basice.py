import numpy as np
import torch

import unittest

class TestBlocklocalScope(unittest.TestCase):
    def setUp(self):
        self.x = torch.ones([2,4])
        self.y = torch.ones([2,4]) * 2
        pass
    
    """
    Under this case, type hint of 'flag' is needed,
    otherwise it will raise error. Because Torch will
    infer type(flag) as Tensor. The error msg as follows:

    RuntimeError: 
        Expected a default value of type Tensor (inferred) on parameter "flag".Because "flag" was not annotated with an explicit type it is assumed to be type 'Tensor'.:
    """
    def test_scope(self):
        @torch.jit.script
        def func(x, y, flag:bool=True): 
            bs = x.shape[0]
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
        
        out = func(self.x, self.y)
        print(out)

        # see IR
        print(func.code)
        # see graph
        print(func.graph)
        """
        A prim::If is generated in Graph, it contains two blocks: block(0), block(1).
        Each block has outputs with same length, it's designed on purpose.
        """
        
    
if __name__ == "__main__":
    unittest.main()