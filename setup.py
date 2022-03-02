from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("CASSPER2.py"),
)

# Compile:
# python setup.py build_ext --inplace
# or
# python setup.py build_ext --force --user
# python setup.py install --user

