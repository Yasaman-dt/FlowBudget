"""Shared solver budget laws used by probe-sensitivity analysis."""
import math

NFE_PER_STEP = {'euler': 1, 'euler-maruyama': 1, 'heun': 2}
FORMULAS = {'euler': ('R_theta', 0.5, 1), 'heun': ('R_theta_heun', 1 / 12, 2)}


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def predict_budget(c_hat, kappa, gamma, tau):
    """Exact prescribed power law; K counts solver steps, without sweep clipping."""
    for name, value in [('C_hat', c_hat), ('kappa', kappa), ('gamma', gamma), ('tau', tau)]:
        positive(value, name)
    value = (kappa * c_hat / tau) ** (1 / gamma)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Budget overflow/underflow; revise numeric search bounds')
    return math.ceil(value)

