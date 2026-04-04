"""Command-line entry point for the skating aerial alignment project."""

from skating_aerial_alignment.modeling import SkaterFlightBiomod


def main() -> None:
    """Print the neutral-pose principal moments of inertia."""

    moments = SkaterFlightBiomod().principal_moments()
    print("Principal moments of inertia [kg.m^2]:")
    print(f"I1={moments[0]:.4f}, I2={moments[1]:.4f}, I3={moments[2]:.4f}")


if __name__ == "__main__":
    main()
