from Modules.sympy.classes import *
from typing import Union
from sympy.core.numbers import Integer, Float, ImaginaryUnit, One, Half, Rational
from sympy.core.power import Pow
from sympy.core.expr import Expr
from sympy import eye, kronecker_product, Mul, Add, Abs, diag, symbols, Matrix
from sympy.physics.quantum import Commutator

from numpy import any as np_any
from numpy import sum as np_sum
from numpy import all as np_all
from multimethod import multimethod

numbers_list = [int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]



@multimethod
def get_terms_and_factors(expr: Union[Operator, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """Returns tuple of two lists: one containing ordered terms within expr, the second 
        containing lists of factors for each term."""
    return [expr], [[expr]]

@multimethod
def get_terms_and_factors(expr: Pow):
    """Returns tuple of two lists: one containing ordered terms within expr, the second 
        containing lists of factors for each term."""
    pow_base, pow_exp = expr.as_base_exp()
    return [expr], [[pow_base for _ in range(pow_exp)]]

@multimethod
def get_terms_and_factors(expr : Expr):
    """Returns tuple of two lists: one containing ordered terms within expr, the second 
        containing lists of factors for each term."""
    expr = expr.expand()
    terms = expr.as_ordered_terms()
    factors_of_terms = []
    for term in terms:
        factors = term.as_ordered_factors()
        factors_list = []
        for f in factors:
            _, f_list = get_terms_and_factors(f)
            factors_list += f_list[0]
        factors_of_terms.append(factors_list)

    return terms, factors_of_terms

@multimethod
def group_by_order(expr: RDsymbol):
    """Returns dict of expressions separated into orders.
    """
    return expr.order

@multimethod
def group_by_order(expr: Union[RDBoson, RDOperator, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    return 0

@multimethod
def group_by_order(expr: Expr):
    """Returns dict of expressions separated into orders."""
    terms, factors_of_terms = get_terms_and_factors(expr)
    
    order_separated = {}
    
    for term, factors in zip(terms, factors_of_terms):
        orderr = np_sum(group_by_order(factor) for factor in factors)
        order_separated[orderr] = order_separated.get(orderr, 0) + term
        
    return order_separated

@multimethod
def group_by_infinite(expr: Union[RDBoson, RDOperator, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """Returns dict of expressions separated into infinite and finite subspaces."""
    return isinstance(expr, RDBoson)

@multimethod
def group_by_infinite(expr: Expr):
    """Returns dict of expressions separated into infinite and finite subspaces."""
    terms, factors_of_terms = get_terms_and_factors(expr)

    infinity_separated = dict()

    for term, factors in zip(terms, factors_of_terms):
        infinity = np_any([group_by_infinite(factor) for factor in factors])
        infinity_separated[infinity] = infinity_separated.get(infinity, 0) + term

    return infinity_separated

@multimethod
def count_bosons(expr: Union[RDOperator, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """Returns number of bosons in expression."""
    return None

@multimethod
def count_bosons(expr: RDBoson):
    """Returns number of bosons in expression."""
    return expr.subspace , {"annihilation" if expr.is_annihilation else "creation": 1}

@multimethod
def count_bosons(expr: Expr):
    """Returns number of bosons in expression."""
    terms, factors_of_terms = get_terms_and_factors(expr)

    boson_count = dict()
    for term in factors_of_terms:
        for factor in term:
            result_count = count_bosons(factor)
            if result_count is  None:
                continue
            subspace, result_count = result_count
            for key, item in result_count.items():
                if boson_count.get(subspace) is None:
                    boson_count[subspace] = {key: item}
                    continue
                boson_count.get(subspace).update({key: boson_count.get(subspace, dict()).get(key, 0) + item})

    return boson_count



@multimethod
def group_by_diagonal(expr: RDOperator):
    """Returns dict of expressions separated into diagonal and non-diagonal terms."""
    return {np_sum(Abs(expr.matrix - diag(*expr.matrix.diagonal()))) == 0 : expr}

@multimethod
def group_by_diagonal(expr: Union[RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """Returns dict of expressions separated into diagonal and non-diagonal terms."""
    return {True : expr}

@multimethod
def group_by_diagonal(expr: Union[Expr, RDBoson]):
    """Returns dict of expressions separated into diagonal and non-diagonal terms."""
    terms, factors_of_terms = get_terms_and_factors(expr)

    diagonal_separated = dict()

    for term, factors_of_term in zip(terms, factors_of_terms):
        bosons_counts = count_bosons(term)
        is_boson_diagonal = np_all([boson_count.get("creation", 0) == boson_count.get("annihilation", 0) for boson_count in bosons_counts.values()])
        is_finite_diagonal = np_all([list(group_by_diagonal(factor).keys())[0] for factor in factors_of_term if not isinstance(factor, RDBoson)])

        diagonal_separated[is_boson_diagonal and is_finite_diagonal] = diagonal_separated.get(is_boson_diagonal and is_finite_diagonal, 0) + term

    return diagonal_separated


@multimethod
def get_finite_identities(expr: Union[RDBoson, RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    """Returns dict of sympy matrices for each subspace spanned by expr."""
    return dict()

@multimethod
def get_finite_identities(expr: RDOperator):
    return {expr.subspace : eye(expr.dim)}

@multimethod
def get_finite_identities(expr: Expr):
    """Returns dict of sympy matrices for each subspace spanned by expr."""
    terms, factors_of_terms = get_terms_and_factors(expr)

    identities = dict()
    for term in factors_of_terms:
        for factor in term:
            identities.update(get_finite_identities(factor))
    return identities



@multimethod
def domain_expansion(expr: Union[RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational], subspaces=None, identities=None):
    """Returns sympy Matrix of expanded expression.
    expr is expression to expand.
    subs_dict is dictionary containing substitution rules for each operator into matrix form.
    subspaces is ordered list of subspaces (order will indicate in which order to perform kronecker_product).
    identities is list of identity operators in each subspace.
    """
    return expr

@multimethod
def domain_expansion(expr: RDBoson, subspaces, identities=None):
    """Returns sympy Matrix of expanded expression.
    expr is expression to expand.
    subs_dict is dictionary containing substitution rules for each operator into matrix form.
    subspaces is ordered list of subspaces (order will indicate in which order to perform kronecker_product).
    identities is list of identity operators in each subspace.
    """

    identities = get_finite_identities(expr) if identities is None else identities
    
    matrices = [identities[subspace] for subspace in subspaces]
    return kronecker_product(*matrices) * expr

@multimethod
def domain_expansion(expr: RDOperator, subspaces, identities=None):
    """Returns sympy Matrix of expanded expression.
    expr is expression to expand.
    subs_dict is dictionary containing substitution rules for each operator into matrix form.
    subspaces is ordered list of subspaces (order will indicate in which order to perform kronecker_product).
    identities is list of identity operators in each subspace.
    """
    identities = get_finite_identities(expr) if identities is None else identities
    matrices = [identities[subspace] if subspace != expr.subspace else expr.matrix for subspace in subspaces]
    return kronecker_product(*matrices)

@multimethod        
def domain_expansion(expr:Expr, subspaces, identities=None):
    """Returns sympy Matrix of expanded expression.
    expr is expression to expand.
    subs_dict is dictionary containing substitution rules for each operator into matrix form.
    subspaces is ordered list of subspaces (order will indicate in which order to perform kronecker_product).
    identities is list of identity operators in each subspace.
    """

    identities = get_finite_identities(expr) if identities is None else identities
    terms, factors_of_terms = get_terms_and_factors(expr)
    
    matrices = []
    for term in factors_of_terms: # "Can we avoid all these for loops?"
        matrices.append(Mul(*[domain_expansion(factor, subspaces, identities) for factor in term]))
    return Add(*matrices)

def nested_commutator(A, B, k=1):
    if k == 0:
        return A
    if k == 1:
        return Commutator(A, B)

    return Commutator(nested_commutator(A, B, k-1), B)

def group_by_infinite_operators(expr):
    infinit_dict = group_by_infinite(expr)
    result_dict = {1: infinit_dict.get(False, 0)}

    infinite_expr = infinit_dict.get(True, 0)
    terms, factors_of_terms = get_terms_and_factors(infinite_expr)

    for term in factors_of_terms:
        result_infinite_term = 1
        result_finite_term = 1
        for factor in term:
            if isinstance(factor, RDBoson):
                result_infinite_term *= factor
                continue
            result_finite_term *= factor
        result_dict[result_infinite_term] = result_dict.get(result_infinite_term, 0) + result_finite_term
    
    return result_dict


def S(dim, name, order = 1):
    symbols_list = [RDsymbol(name + f"_{i}", order=order) for i in range(dim**2)]
    mat_representation =  Matrix([symbols_list[i] for i in range(dim**2)]).reshape(dim, dim)

    S_op  = RDOperator(name, subspace = "finite", matrix=mat_representation, dim=dim)
    #S_op.subspace = "finite"
    return S_op

def expand_commutator(expr):
    expr_expanded = expr.expand(commutator=True)
    while expr_expanded != expr:
        expr = expr_expanded
        expr_expanded = expr.expand(commutator=True)
    return expr_expanded

def get_finite_operators(names, subspaces, dims, matrices):
    operators = [RDOperator(names[i], subspace = subspaces[i], dim = dims[i], matrix=matrices[i]) for i in range(len(names))] 
    identities = get_finite_identities(np_sum(operators))
    dim = Mul(*list(dict(zip(subspaces, dims)).values()))
    expanded_operators = [kronecker_product(*[identities[symbols(subspace)] if symbols(subspace) != op.subspace else op.matrix for subspace in subspaces]) for op in operators]
    new_operators = [RDOperator(names[i], subspace = "finite", dim = dim, matrix=expanded_operators[i]) for i in range(len(names))]
    return new_operators


@multimethod
def get_matrix(expr: Union[RDsymbol, int, float, complex, Integer, Float, ImaginaryUnit, One, Half, Rational]):
    print(expr)
    return expr

@multimethod
def get_matrix(expr: RDOperator):
    print(expr.matrix)
    return expr.matrix

@multimethod
def get_matrix(expr : Expr):
    "Only use expr without infinite operators"
    terms, factors_of_terms = get_terms_and_factors(expr)
    matrices = []
    for term in factors_of_terms:
        mat_res = 1
        for factor in term:
            mat_res *= get_matrix(factor)
        matrices.append(mat_res)
    return Add(*matrices)