from setuptools import setup, find_packages

setup(
    name="wmcp",
    version="0.1.0",
    packages=find_packages(),
    entry_points={"console_scripts": ["wmcp=wmcp.cli:main"]},
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "numpy", "scipy"],
)
