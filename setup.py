from setuptools import setup, find_packages

setup(
    name="wmcp",
    version="0.1.0",
    description="World Model Communication Protocol — discrete compositional communication between heterogeneous vision models",
    long_description=open("wmcp/README.md").read(),
    long_description_content_type="text/markdown",
    author="Tomek Kaszynski",
    author_email="t.kaszynski@proton.me",
    url="https://github.com/TomekKaszynski/emergent-physics-comm",
    project_urls={
        "Paper": "https://doi.org/10.5281/zenodo.19197757",
        "Protocol Spec": "https://github.com/TomekKaszynski/emergent-physics-comm/tree/main/protocol-spec",
    },
    packages=find_packages(exclude=["wmcp_ros2*", "tests*"]),
    entry_points={"console_scripts": ["wmcp=wmcp.cli:main"]},
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "numpy", "scipy"],
    extras_require={"dev": ["pytest", "build"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
