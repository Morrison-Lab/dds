"""Let reticulate capture chunk output once torch is imported.

reticulate's knitr engine wraps every Python chunk in an output-capture
context that points each logging handler writing to sys.stderr at its
capture buffer and back again. torch 2.14 registers handlers of class
torch._logging._internal._StderrHandler, whose `stream` is a read-only
property returning sys.stderr at call time, so reticulate's assignment
raises "AttributeError: property 'stream' of '_StderrHandler' object has
no setter" and the render stops.

Such a handler already follows sys.stderr, so it needs no redirecting.
This wraps reticulate's `_setStream` to leave exactly those handlers alone
and to behave as before for every other handler. Run it once per page,
before the first chunk that imports torch: `tools/use-torch.R` does.
"""

import rpytools.output as _output


def _set_stream(handler, stream, _original=_output._setStream):
    prop = getattr(type(handler), "stream", None)
    if isinstance(prop, property) and prop.fset is None:
        return handler.stream
    return _original(handler, stream)


if getattr(_output._setStream, "__name__", "") != "_set_stream":
    _output._setStream = _set_stream
