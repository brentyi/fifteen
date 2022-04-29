from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fifteen",
    version="0.0.0",
    description="Training infrastructure for JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/brentyi/fifteen",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="MIT",
    packages=find_packages(),
    package_data={"fifteen": ["py.typed"]},
    python_requires=">=3.8",
    install_requires=[
        "dcargs",
        "flax",
        "jax",
        "jaxlib",
        "jax_dataclasses",
        "termcolor",
        "tensorflow",  # Needed for Flax serialization/Tensorboard; primarily for `tensorflow.io.gfile`.
        "types-termcolor",
        "typing_extensions",
        "multiprocess",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
