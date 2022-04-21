import setuptools

try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""


setuptools.setup(
    name="gmm_torch",
    version="v0.0.1",
    url="https://github.com/MilesCranmer/gmm-torch",
    install_requires=["numpy", "scipy"],
    packages=setuptools.find_packages(),
)