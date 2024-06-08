from sympy.physics.quantum import Operator, TensorProduct
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.core.sympify import sympify
from sympy.core.singleton import S
from sympy import Basic


class SubSpace(Basic):
    @property
    def name(self):
        return self.args[0]

    @property
    def dimension(self):
        return self.args[1]

    def __new__(cls, name, dim):
        return Basic.__new__(cls,  name, dim)


class RDOperator(Operator):
    @property
    def name(self):
        return self.args[0]

    @property
    def subspace(self):
        return self.args[1]

    def _sympystr(self, printer):
        return printer._print(self.name)

    def _pretty(self, printer):
        return self._sympystr(printer)

    def _latex(self, printer):
        return self._sympystr(printer)

    def __new__(cls, name, subspace: SubSpace):
        return Operator.__new__(cls, name, subspace)

    def _eval_commutator_RDOperator(self, other, **options):
        if self.subspace != other.subspace:
            return S.Zero
        return self * other - other * self

    def _eval_commutator_BosonOp(self, other, **options):
        return S.Zero

    def _eval_commutator_RDsymbol(self, other, **options):
        return S.Zero


class RDsymbol(Operator):
    @property
    def name(self):
        return self.args[0]

    @property
    def order(self):
        return self.args[1]

    def _sympystr(self, printer):
        return printer._print(f"{printer._print(self.name)}_{printer._print(self.order)}")

    def _pretty(self, printer):
        return self._sympystr(printer)

    def _latex(self, printer):
        return self._sympystr(printer)

    def __new__(cls, name, order=1):
        return Operator.__new__(cls, sympify(name), sympify(order))

    def _eval_commutator_RDOperator(self, other, **options):
        return S.Zero

    def _eval_commutator_BosonOp(self, other, **options):
        return S.Zero
