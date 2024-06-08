from pysw.classes import RDOperator, RDsymbol
from pysw.utils import *
from pysw.solver import *
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy import Pow, Mul, UnevaluatedExpr
import sympy as sp
sp.init_printing()


spin = SubSpace("spin", dim=2)
charge = SubSpace("charge", dim=2)

sigma_x = RDOperator('sigma_x', subspace=spin)
sigma_z = RDOperator('sigma_z', subspace=spin)
tau_x = RDOperator('tau_x', subspace=charge)
a = BosonOp("a")

hbar = RDsymbol("hbar", order=0)
omega = RDsymbol("omega", order=0)
omega_z = RDsymbol("omega_z", order=0)
g = RDsymbol("g", order=1)
f = RDsymbol("f", order=2)

H = sum([
    hbar * omega * Dagger(a) * a**2,
    hbar * omega_z * sp.Rational(1, 2) * sigma_z,
    hbar * g * (a + Dagger(a)) * sigma_x,
    hbar * f * tau_x
])
sp.pprint(H)

group_by_infinite_operaters(H)

H = expand_operators(H, subs=[
    (sigma_x, sp.Matrix([[0, 1], [1, 0]])),
    (sigma_z, sp.Matrix([[1, 0], [0, -1]])),
    (tau_x, sp.Matrix([[0, 1], [1, 0]]))
])
