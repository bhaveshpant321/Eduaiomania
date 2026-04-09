import random

def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamps a value to be within [min_val, max_val]."""
    return max(min_val, min(val, max_val))

def apply_gaussian_noise(val: float, sigma: float = 0.05) -> float:
    """
    Applies Gaussian noise to a value to simulate human capriciousness/fickleness.
    Clamps the result to [0, 1].
    """
    noise = random.gauss(0, sigma)
    # The noise magnitude could optionally scale with 'val' but adding raw noise is straightforward
    return clamp(val + noise)
