from .classes import *
from sympy import eye, Mul, Add, Expr, Pow, UnevaluatedExpr, S
from sympy.physics.quantum import Commutator, TensorProduct
from sympy.physics.quantum.boson import BosonOp
from collections import defaultdict
import sympy as sp


def group_by_order(expr: Expr):
    """Returns dict of expressions separated into orders."""
    result = defaultdict(lambda: S.Zero)
    for term in expr.as_ordered_terms():
        # Sum orders of RDsymbols in term
        order = int(sum(i.order for i in term.find(RDsymbol)))
        result[order] = Add(result[order], term)
    return result


def group_by_infinite(expr: Expr):
    """Returns 2 expressions seperated into infinite and finite terms."""
    result = [S.Zero, S.Zero]  # infinite, finite
    for term in expr.as_ordered_terms():
        # If term contains BosonOp, it is infinite
        idx = int(term.has(BosonOp))
        result[idx] = Add(result[idx], term)
    return result


def group_by_diagonal(expr: Expr):
    """Returns 2 expressions seperated into diagonal and non-diagonal terms."""
    result = [S.Zero, S.Zero]  # diagonal, non-diagonal

    for term in expr.as_ordered_terms():
        # Expand Pow to Mul: a**2 -> a*a
        count_term = term.replace(
            lambda x: x.is_Pow and x.exp > 0 and isinstance(x.base, BosonOp),
            lambda x: UnevaluatedExpr(Mul(*[x.base]*x.exp, evaluate=False))
        )

        # Get all Boson Symbols
        boson_list = set(boson.name for boson in expr.atoms(BosonOp))

        def check_balanced(name):
            return count_term.count(BosonOp(name)) == count_term.count(Dagger(BosonOp(name)))

        # Check count Boson creation and annihalation are equal
        boson_check = all(check_balanced(name) for name in boson_list)

        # Check if all matrices are diagonal
        matrix_check = all(m.is_diagonal()
                           for m in term.find(sp.Matrix))

        idx = int(boson_check and matrix_check)
        result[idx] = Add(result[idx], term)

    return result


def expand_operators(expr: Expr, subs):
    """Expand RDOperators into their Matrix representations, taking into account all subspaces"""

    # Get all subspaces
    subspaces = list(set(op.subspace for op in expr.atoms(RDOperator)))

    for i, sub in enumerate(subs):
        op, mat = sub
        assert isinstance(op, RDOperator)
        assert mat.shape[0] == mat.shape[1]
        assert op.subspace.dimension == mat.shape[0]

        subs[i] = (op, TensorProduct(*[
            mat if op.subspace == space else eye(space.dimension)
            for space in subspaces
        ]))

    return expr.subs(subs)


def group_by_infinite_operaters(expr: Expr):
    result = expr.coeffs(BosonOp("a"))
    return result


def nested_commutator(A, B, k=1):
    if k == 0:
        return A
    if k == 1:
        return Commutator(A, B)
    else:
        return Commutator(nested_commutator(A, B, k-1), B)
