#!/usr/bin/env python3
"""Print the current validated mesh profile for a Greek-particle ADDA size."""

import argparse

from greek_profiles import select_greek_profile


def remove_obj_suffix(name):
    return name[:-4] if name.endswith(".obj") else name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ax", type=float, help="ADDA size parameter A_x")
    parser.add_argument("--command", action="store_true", help="print a BEM command template")
    args = parser.parse_args()

    profile, extrapolated = select_greek_profile(args.ax)
    status = "extrapolated" if extrapolated else "validated"
    print(f"A_x={args.ax:g}")
    print(f"mesh={profile.mesh}")
    print(f"status={status}")
    print(f"note={profile.note}")
    if args.command:
        out = (
            "runs/greek_larger_valid/"
            f"bem_{remove_obj_suffix(profile.mesh.rsplit('/', 1)[-1])}_"
            f"Ax{args.ax:g}_a95b65g20_q4_n181.json"
        )
        print(
            "./bin/bem_cuda_fmm --solver dense --system pmchwt "
            f"--obj {profile.mesh} --ka {args.ax:g} --ri 1.6 0.002 "
            f"--quad 4 --orient 95 65 20 --ntheta 181 --out {out}"
        )


if __name__ == "__main__":
    main()
