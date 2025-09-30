from champy.Hamiltonian import Hamiltonian
from pauliarray import Operator


class PauliHamiltonian(Hamiltonian):

    def __init__(self, operator: Operator):
        self.operator = Operator
