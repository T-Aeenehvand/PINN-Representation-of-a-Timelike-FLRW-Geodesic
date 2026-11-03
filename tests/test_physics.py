import numpy as np
from flrw_geodesic_pinn.physics import conformal_targets, initial_affine_state, peculiar_speed


def test_initial_first_integral_values():
    eta_p, x_p = initial_affine_state()
    assert np.isclose(eta_p, np.sqrt(272.0))
    assert np.isclose(x_p, 16.0)


def test_conformal_targets_initial():
    dx, dl = conformal_targets(0.5)
    assert np.isclose(dx, 1.0 / np.sqrt(1.0625))
    assert np.isclose(dl, 0.0625 / np.sqrt(1.0625), rtol=1e-12)


def test_peculiar_speed_decreases():
    assert peculiar_speed(1.5) < peculiar_speed(0.5)
