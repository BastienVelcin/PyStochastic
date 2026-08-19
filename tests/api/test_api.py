"""
Tests for the public PyStochastic API.
"""


def test_main_package_imports():
    import pystochastic

    assert pystochastic is not None


def test_distribution_package_imports():
    import pystochastic.dist

    assert pystochastic.dist is not None


def test_montecarlo_package_imports():
    import pystochastic.montecarlo

    assert pystochastic.montecarlo is not None


def test_process_package_imports():
    import pystochastic.processes

    assert pystochastic.processes is not None


def test_sde_package_imports():
    import pystochastic.sde

    assert pystochastic.sde is not None


def test_public_distribution_classes():
    from pystochastic.dist import (
        Distribution,
        DiscreteDistribution,
    )

    assert Distribution is not None
    assert DiscreteDistribution is not None


def test_public_montecarlo_classes():
    from pystochastic.montecarlo import (
        MonteCarlo,
        MonteCarloProcess,
    )

    assert MonteCarlo is not None
    assert MonteCarloProcess is not None