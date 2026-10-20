"""Safe evaluation of user expressions (COMSOL-style parameters and functions).

Every numeric entry of a model may be a number or a string expression such as
``"V0*min(t/1e-9, 1)"``.  Expressions can use global parameters, physical
constants (c, e, me, eps0, ...), numpy math functions and, depending on the
context, the variables ``t`` (time), ``z``, ``y``, ``r`` (coordinates).
"""
from __future__ import annotations

import ast
import math
import numpy as np

from . import constants as C

_ALLOWED_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
    "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs, "tanh": np.tanh,
    "sinh": np.sinh, "cosh": np.cosh, "arcsin": np.arcsin, "arccos": np.arccos,
    "arctan": np.arctan, "arctan2": np.arctan2, "minimum": np.minimum,
    "maximum": np.maximum, "where": np.where, "heaviside": lambda x: np.heaviside(x, 1.0),
    "step": lambda x: np.heaviside(x, 1.0), "sign": np.sign, "floor": np.floor,
    "ceil": np.ceil, "clip": np.clip, "pi": math.pi, "inf": math.inf,
    "gauss": lambda x, x0, s: np.exp(-0.5 * ((x - x0) / s) ** 2),
    "rect": lambda x, a, b: ((x >= a) & (x <= b)).astype(float),
    "ramp": lambda t, tr: np.clip(t / tr, 0.0, 1.0) if tr > 0 else np.ones_like(t),
    "linspace": np.linspace, "arange": np.arange, "array": np.array,
    "logspace": np.logspace,
}


def _min(*a):
    if len(a) == 1:
        return np.min(a[0])
    out = a[0]
    for b in a[1:]:
        out = np.minimum(out, b)
    return out


def _max(*a):
    if len(a) == 1:
        return np.max(a[0])
    out = a[0]
    for b in a[1:]:
        out = np.maximum(out, b)
    return out


_ALLOWED_FUNCS["min"] = _min
_ALLOWED_FUNCS["max"] = _max

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.FloorDiv, ast.USub, ast.UAdd, ast.Compare, ast.Lt, ast.LtE, ast.Gt,
    ast.GtE, ast.Eq, ast.NotEq, ast.BoolOp, ast.And, ast.Or, ast.IfExp,
    ast.Tuple, ast.List, ast.keyword, ast.BitAnd, ast.BitOr, ast.Not,
)


class ExpressionError(ValueError):
    pass


def _check(tree):
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"Disallowed syntax in expression: {type(node).__name__}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ExpressionError("Only plain function calls are allowed")


_cache: dict[str, object] = {}


def compile_expr(src: str):
    src = str(src).strip()
    code = _cache.get(src)
    if code is None:
        try:
            tree = ast.parse(src, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Syntax error in '{src}': {exc.msg}") from None
        _check(tree)
        code = compile(tree, "<expr>", "eval")
        _cache[src] = code
    return code


class Evaluator:
    """Evaluates expressions with a parameter table."""

    def __init__(self, parameters: dict | None = None):
        self.base = dict(_ALLOWED_FUNCS)
        self.base.update(C.NAMESPACE)
        self.params: dict[str, float] = {}
        self.set_parameters(parameters or {})

    # ------------------------------------------------------------------
    def set_parameters(self, parameters: dict):
        """Resolve parameters which may reference each other (any order)."""
        pending = {k: v for k, v in parameters.items() if str(k).strip()}
        resolved: dict[str, float] = {}
        for _ in range(len(pending) + 1):
            progress = False
            for k in list(pending):
                ns = dict(self.base)
                ns.update(resolved)
                try:
                    val = eval(compile_expr(pending[k]), {"__builtins__": {}}, ns)
                except NameError:
                    continue
                resolved[k] = val
                del pending[k]
                progress = True
            if not pending or not progress:
                break
        if pending:
            name = next(iter(pending))
            raise ExpressionError(f"Cannot resolve parameter '{name}' = '{pending[name]}'")
        self.params = resolved

    def namespace(self, **extra):
        ns = dict(self.base)
        ns.update(self.params)
        ns.update(extra)
        return ns

    # ------------------------------------------------------------------
    def value(self, x, **variables):
        """Evaluate ``x`` (number, list or expression string) to a number/array."""
        if x is None:
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        if isinstance(x, (list, tuple)):
            return [self.value(v, **variables) for v in x]
        if isinstance(x, np.ndarray):
            return x
        s = str(x).strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            pass
        try:
            return eval(compile_expr(s), {"__builtins__": {}}, self.namespace(**variables))
        except ExpressionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExpressionError(f"Error evaluating '{s}': {exc}") from None

    def num(self, x, default=None, **variables) -> float:
        v = self.value(x, **variables)
        if v is None:
            if default is None:
                raise ExpressionError("Missing numeric value")
            return float(default)
        return float(v)

    def is_time_dependent(self, x) -> bool:
        if isinstance(x, (int, float)) or x is None:
            return False
        try:
            tree = ast.parse(str(x), mode="eval")
        except SyntaxError:
            return False
        return any(isinstance(n, ast.Name) and n.id == "t" for n in ast.walk(tree))

    def func_t(self, x, default=0.0):
        """Return f(t) for an expression in time."""
        if x is None or (isinstance(x, str) and x.strip() == ""):
            d = float(default)
            return lambda t: d
        if not self.is_time_dependent(x):
            v = self.num(x)
            return lambda t: v
        code = compile_expr(str(x))
        ns = self.namespace()

        def f(t):
            ns["t"] = t
            return float(eval(code, {"__builtins__": {}}, ns))
        return f

    def func_space(self, x, names=("z", "y", "r", "t")):
        """Return f(z, y, t) for a spatial expression (y doubles as r)."""
        code = compile_expr(str(x))
        ns = self.namespace()

        def f(z, y, t=0.0):
            ns.update(z=z, y=y, r=y, x=y, t=t)
            out = eval(code, {"__builtins__": {}}, ns)
            return np.broadcast_to(np.asarray(out, dtype=float), np.broadcast(z, y).shape).copy()
        return f
