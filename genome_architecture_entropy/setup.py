from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

# define an extension that will be cythonized and compiled
ext = Extension(name="cosegregation_internal", sources=["cosegregation_internal.pyx"])
setup(ext_modules=cythonize(ext), include_dirs=[numpy.get_include()])