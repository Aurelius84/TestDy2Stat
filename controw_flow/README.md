# Test for ControlFlow

In Python, builtin control-flow syntax contains:

+ if
+ for
+ while


## TensorFlow

### IF

In TensorFlow, if will be converted into the following format: 
```python
def if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts):
    """
    The state is represented by the variable `x`. The `body, `orelse` and
    `set_state` functions must bind to the original `x` symbol, using `nonlocal`.
    """
  if tensors.is_dense_tensor(cond):
    _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
  else:
    _py_if_stmt(cond, body, orelse)   # no need get_state, set_state

```

For Example:
```python
"""
def func(x, y):
    avg_x = tf.reduce_mean(x)
    if avg_x > 10:
        out = y + 1
        x = out * 2
    else:
        out = x + 1
        y = out * 2

    return x, y, out
"""
def tf__func(x, y):
    with ag__.FunctionScope('func', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
        do_return = False
        retval_ = ag__.UndefinedReturnValue()
        avg_x = ag__.converted_call(ag__.ld(tf).reduce_mean, (ag__.ld(x),), None, fscope)

        def get_state():
            return (out, x, y)

        def set_state(vars_):
            nonlocal x, out, y
            (out, x, y) = vars_

        def if_body():
            nonlocal x, out, y
            out = (ag__.ld(y) + 1)
            x = (ag__.ld(out) * 2)

        def else_body():
            nonlocal x, out, y
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
```


## Early Return
