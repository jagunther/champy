import pytest

from champy.fermion2qubit import h2e_check_convention, h2e_order


@pytest.mark.parametrize("integrals_random", [8], indirect=True)
def test_check_h2e_convention_chemist(integrals_random):
    assert h2e_check_convention(integrals_random[2]) == "chemist"


@pytest.mark.parametrize("integrals_random_physicist", [8], indirect=True)
def test_check_h2e_convention_physicist(integrals_random_physicist):
    assert h2e_check_convention(integrals_random_physicist[2]) == "physicist"


@pytest.mark.parametrize("integrals_random", [8], indirect=True)
def test_h2e_order_chemist(integrals_random):
    h2e = integrals_random[2]
    assert h2e_check_convention(h2e_order(h2e, "chemist")) == "chemist"
    assert h2e_check_convention(h2e_order(h2e, "physicist")) == "physicist"


@pytest.mark.parametrize("integrals_random_physicist", [8], indirect=True)
def test_h2e_order_physicist(integrals_random_physicist):
    h2e = integrals_random_physicist[2]
    assert h2e_check_convention(h2e_order(h2e, "chemist")) == "chemist"
    assert h2e_check_convention(h2e_order(h2e, "physicist")) == "physicist"
