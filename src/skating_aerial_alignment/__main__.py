"""Command-line entry point for the skating aerial alignment project."""

from skating_aerial_alignment.visualization import launch_app


def main() -> None:
    """Launch the interactive GUI."""

    launch_app()


if __name__ == "__main__":
    main()
