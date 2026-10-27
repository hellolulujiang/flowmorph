"""Three maths routines for the numba kernels.

``sincos``     the sine and the cosine of one angle: on macOS from one call of ``__sincos_stret``, elsewhere from the
               separate ``sin`` and ``cos``.  Their last bits can differ from one system to another, and with them a
               tie of the plan form (which pixel is farthest, on which side of a line a pixel centre lies).
``remainder``  the IEEE remainder of the system's maths library (a longitude kept next to another, whatever turn it
               is on).
``fma``        the fused multiply-add, for the exact sign of an orientation in the convex hull.
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
        """(sin x, cos x) from one ``__sincos_stret`` call."""
        return _sincos_stret(float(x))
else:
    @njit(inline="always")
    def sincos(x):
        """(sin x, cos x) from the separate ``sin`` and ``cos``."""
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
