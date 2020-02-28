from setuptools import find_packages, setup

setup(
    name="maskdet",
    version=0.1,
    author="rainweic",
    url="None",
    description="mask det",
    packages=find_packages(exclude=("model", "trainfacecla")),
    python_requires=">=3.5",
    install_requires=[
        "mxnet",
        "opencv-python"
    ],
)