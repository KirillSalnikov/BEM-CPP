"""Validated Greek-particle mesh profiles for ADDA comparisons."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GreekProfile:
    max_ax: float
    mesh: str
    note: str


PROFILES = [
    GreekProfile(
        15.68,
        "runs/greek_larger_valid/meshes/shapeafine_res_f3400_ag8.obj",
        "fast validated profile for A_x <= 15.68",
    ),
    GreekProfile(
        18.94,
        "runs/greek_larger_valid/meshes/shapeafine_res_f4200_ag8.obj",
        "validated profile for 17.19 <= A_x <= 18.94",
    ),
    GreekProfile(
        20.76,
        "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f3400_a35.obj",
        "best strict score at A_x=20.76",
    ),
    GreekProfile(
        25.09,
        "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f4200_a35.obj",
        "validated through A_x=25.09",
    ),
    GreekProfile(
        27.50,
        "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f6000_a35.obj",
        "faster than a45 and within 15% of the best strict score at A_x=27.5",
    ),
    GreekProfile(
        30.25,
        "runs/greek_larger_valid/meshes/shapeafine_res_f4200_closed_gmsh_f7000_a45.obj",
        "best available at A_x=30.25; strict S12 remains the limiting error",
    ),
]


def select_greek_profile(ax):
    ax = float(ax)
    for profile in PROFILES:
        if ax <= profile.max_ax + 1e-12:
            return profile, False
    return PROFILES[-1], True
