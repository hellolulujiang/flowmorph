"""Three routines of the C library, called as the C code of FlowMorph calls them.

The plan form of a catchment (:mod:`flowmorph.sphere`) is decided by signs and ties: which
pixel is farthest, on which side of a line a pixel centre lies.  To give the C's answers the port has to compute
the same doubles, and three of the C's routines have no equal in numba:

``sincos``     the sine and the cosine of one angle.  On macOS clang compiles ``cos(x)`` and ``sin(x)`` of one
               argument into one call of ``__sincos_stret``, whose last bit can differ from the two separate
               routines; the C's tables of the rows' latitudes and the columns' longitudes are made that way, and
               so are these.  On other systems the two separate routines are called, as a C compiled there does.
``remainder``  the IEEE remainder of the C library (a longitude kept next to another, whatever turn it is on).
``fma``        the fused multiply-add, which the C calls by name in one place: the exact sign of an orientation in
               the convex hull.  Everywhere else the C is compiled without fused multiply-adds, and numba does not
               fuse either.
"""

import math
import platform

from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

ON_MACOS = platform.system() == "Darwin"


@intrinsic
def _sincos_stret(typingctx, x):
    if not isinstance(x, types.Float):
        return None
    signature = types.UniTuple(types.float64, 2)(types.float64)

    def codegen(context, builder, signature, args):
        double = ir.DoubleType()
        function_type = ir.FunctionType(ir.LiteralStructType([double, double]), [double])
        function = cgutils.get_or_insert_function(builder.module, function_type, "__sincos_stret")
        pair = builder.call(function, [args[0]])
        return context.make_tuple(builder, signature.return_type,
                                  [builder.extract_value(pair, 0), builder.extract_value(pair, 1)])
    return signature, codegen


@intrinsic
def _remainder(typingctx, x, y):
    if not (isinstance(x, types.Float) and isinstance(y, types.Float)):
        return None
    signature = types.float64(types.float64, types.float64)

    def codegen(context, builder, signature, args):
        double = ir.DoubleType()
        function = cgutils.get_or_insert_function(builder.module, ir.FunctionType(double, [double, double]),
                                                  "remainder")
        return builder.call(function, [args[0], args[1]])
    return signature, codegen


@intrinsic
def _fma(typingctx, x, y, z):
    if not (isinstance(x, types.Float) and isinstance(y, types.Float) and isinstance(z, types.Float)):
        return None
    signature = types.float64(types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        double = ir.DoubleType()
        function = cgutils.get_or_insert_function(builder.module, ir.FunctionType(double, [double, double, double]),
                                                  "llvm.fma.f64")
        return builder.call(function, [args[0], args[1], args[2]])
    return signature, codegen


if ON_MACOS:
    @njit(inline="always")
    def sincos(x):
        """(sin x, cos x) from one ``__sincos_stret`` call, as the C binary on macOS takes them."""
        return _sincos_stret(float(x))
else:
    @njit(inline="always")
    def sincos(x):
        """(sin x, cos x) from the two routines of the C library."""
        return math.sin(x), math.cos(x)


@njit(inline="always")
def remainder(x, y):
    """The IEEE remainder of x by y: x - n y, n the whole number nearest x / y (ties to even)."""
    return _remainder(float(x), float(y))


@njit(inline="always")
def fma(x, y, z):
    """x * y + z rounded once."""
    return _fma(float(x), float(y), float(z))


__all__ = ["sincos", "remainder", "fma", "ON_MACOS"]
