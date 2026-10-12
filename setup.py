from setuptools import setup, find_packages

setup(
    name="hal9000-inside",
    version="0.2.0",
    description="HAL 9000 Inside - Ubuntu Voice Assistant",
    license="AGPL-3.0",
    author="cgegout",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "hal9000-inside = hal9000_inside.cli.main:main",
        ],
    },
)
