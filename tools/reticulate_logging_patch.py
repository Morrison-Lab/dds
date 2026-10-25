"""Let reticulate capture chunk output once torch is imported.

reticulate's knitr engine wraps every Python chunk in an output-capture
context that points each logging handler writing to sys.stdout or
sys.stderr at its capture buffer, and back again afterwards. torch 2.14
registers handlers of class torch._logging._internal._StderrHandler, whose
`stream` is a read-only property returning sys.stderr at call time, so
reticulate's assignment raises "AttributeError: property 'stream' of
'_StderrHandler' object has no setter" and the render stops.

Such a handler writes wherever sys.stderr points, and reticulate has
already pointed sys.stderr at the stream it wants, so the handler's stream
is already the target. This wraps reticulate's `_setStream` to skip exactly
that case: a handler whose `stream` is read-only and already the target.
Every other handler goes to reticulate's own `_setStream` as before, so a
read-only handler pointing anywhere else still raises rather than escaping
the capture. Run it once per page, before the first chunk that imports
torch: `tools/use-torch.R` does.
"""

import rpytools.output as _output


def _set_stream(handler, stream, _original=_output._setStream):
    prop = getattr(type(handler), "stream", None)
    read_only = isinstance(prop, property) and prop.fset is None
    if read_only and handler.stream is stream:
        return stream
    return _original(handler, stream)


_set_stream.skips_read_only_streams = True

if not getattr(_output._setStream, "skips_read_only_streams", False):
    _output._setStream = _set_stream
