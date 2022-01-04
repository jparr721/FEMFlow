from typing import Tuple, Union, List
import numpy as np


def parse_orthotropic_material_coefficients(
    coefficients: Tuple[str, str, str]
) -> np.ndarray:
    if len(coefficients) != 3:
        raise ValueError(f"Orthotropic coefficients require 3 values")

    def check_vals(value: str, n: int = 3) -> Union[List[str], None]:
        values = value.split(",")
        if len(values) != n:
            return None
        else:
            return values

    e, v, g = coefficients
    try:
        e_vals = check_vals(e)
        if e_vals is None:
            raise ValueError("Orthotropic coefficients require 3 youngs modulus'")
        youngs_modulus = np.array(map(float, e_vals))

        v_vals = check_vals(v, 6)
        if v_vals is None:
            raise ValueError("Orthotropic coefficients require 6 poissons ratios")
        poissons_ratio = np.array(map(float, v_vals))

        g_vals = check_vals(g)
        if g_vals is None:
            raise ValueError("Orthotropic coefficients require 3 shear modulus'")
        shear_modulus = np.array(map(float, g_vals))

        return np.array((*youngs_modulus, *poissons_ratio, *shear_modulus))

    except Exception as e:
        raise RuntimeError("Parsing failed") from e


def parse_isotropic_material_coefficients(coefficients: Tuple[str, str]) -> np.ndarray:
    if len(coefficients) != 2:
        raise ValueError(f"Isotropic coefficients require 2 values")

    e, v = coefficients
    try:
        youngs_modulus = float(e)
        poissons_ratio = float(v)
        return np.array((youngs_modulus, poissons_ratio))
    except Exception as e:
        raise RuntimeError("Parsing failed") from e
