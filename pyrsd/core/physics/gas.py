"""
pyrsd/core/physics/gas.py
lookup table for constants for common gases.
"""

# Gladstone-Dale constants m^3/kg
GLADSTONE_DALE: dict[str, float] = {
    "air":      2.23e-4,
    "co2":      4.51e-4,
    "nitrogen": 2.38e-4,
    "helium":   1.96e-5,
    "hydrogen": 1.55e-4,
}

# specific gas constants J/(kg·K)
GAS_CONSTANT: dict[str, float] = {
    "air":      287.058,
    "co2":      188.9,
    "nitrogen": 296.8,
    "helium":   2077.0,
    "hydrogen": 4124.0,
}

def gladstone_dale(gas: str) -> float:
    """looks up for gladstone dale constant for the specified gas in dataset"""
    if gas.lower() not in GLADSTONE_DALE:
        raise ValueError(f"Gladstone-Dale constant not available for '{gas}'. "
                                f"Available gases: {list(GLADSTONE_DALE)}")
    return GLADSTONE_DALE[gas.lower()]

def gas_constant(gas: str) -> float:
    """looks up for specific gas constant value for the specified gas in the dataset"""
    if gas.lower() not in GAS_CONSTANT:
        raise ValueError(f"Specific Gas constant not available for '{gas}'."
                         f"Available gases: {list(GAS_CONSTANT)}")
    return GAS_CONSTANT[gas.lower()]