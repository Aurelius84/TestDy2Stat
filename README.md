# TestDy2Stat
More Unittests for Deep Learning Framework in Dynamic2Static Feature.

## What's this?

## Goals
This repository aims to deeply explore the mechanism of them by writing many unittests to see how they implement this detaily step by step.

## TensorFlow
`@tf.function` is an important feature for TF 2.x and it support transform users' imperative model codes into Static Graph for offline predicting.
### Version

+ tensorflow: `2.8.0`
+ tensorflow-io-gcs-filesystem: `0.24.0`

## PyTorch
Torch introduces `@jit.trace` and `@jit.script` for users, which exports models as `jit::ScriptModule` to be easily loaded by libtorch.

![image](https://user-images.githubusercontent.com/9301846/169251540-c748e4a0-9380-4fed-806e-35d47a1c6bc9.png)


### Mechanism

#### Source to Ast

```python
# obj is source function
ast = get_jit_def(obj, obj.__name__) # <torch._C._jit_tree_views.Def>
```

** Parse Closure**
Torch will parse decorated function closure information and customize lookup routine for variable.
```python
def get_closure(fn):
    """
    Get a dictionary of closed over variables from a function
    """
    captures = {}
    captures.update(fn.__globals__)

    for index, captured_name in enumerate(fn.__code__.co_freevars):
        captures[captured_name] = fn.__closure__[index].cell_contents

    return captures

class closure_lookup(object):
    # This is a class since `closure` is a dict and it's easier in
    # `env_helper` if everything just works with `getattr` calls
    def __getattr__(self, key):
        if key in closure:
            return closure[key]
        elif hasattr(typing, key):
            return getattr(typing, key)
        elif hasattr(builtins, key):
            return getattr(builtins, key)
        return None
```

### Jit Compile

```python
# <torch.jit.ScriptFunction>
fn = torch._C._jit_script_compile(
    qualified_name, ast, _rcb, get_default_args(obj)
)
```
The attributes of `ScriptFunction` contains:
+ code
+ graph
+ schema
  
The methods of `ScriptFunction` contains:
+ save
+ graph_for
+ get_debug_state
+ \_\_call__  # Attention It.

### Version

+ torch: `1.11.0`
+ torchaudio: `0.11.0`
+ torchvision: `0.12.0`
