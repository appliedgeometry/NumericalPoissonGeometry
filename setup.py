import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='numericalpoissongeometry',
    version='1.0.0',
    author="Miguel Evangelista-Alvarado, Jose C. Crispín Ruíz, Pablo Suárez-Serrato",
    author_email="miguel.eva.alv@gmail.com, jcpanta@im.unam.mx, pablo@im.unam.mx",
    license="MIT",
    description="A Python Numeric module for (local) calculus on Poisson manifolds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/appliedgeometry/NumericalPoissonGeometry",
    packages=setuptools.find_packages(),
    install_requires=['sympy', 'numpy', 'torch', 'tensorflow', 'poissongeometry', 'permutation', 'scipy'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.6',
)
