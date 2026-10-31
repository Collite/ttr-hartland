from importlib.metadata import version

import hartland_sim


def test_version_enters_the_platform_line_at_0_11_0() -> None:
    assert hartland_sim.__version__ == "0.11.0"


def test_the_installed_distribution_carries_the_package_version() -> None:
    assert version("hartland-sim") == hartland_sim.__version__
