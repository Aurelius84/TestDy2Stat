"""
# this is a decorated fn, and we need to the underlying fn and its rcb
if hasattr(obj, "__script_if_tracing_wrapper"):
    obj = obj.__original_fn
    _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
"""