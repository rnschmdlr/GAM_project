import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).

cimport numpy as np
cimport cython

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.

DTYPE = int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.

ctypedef np.int_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def dprime_2d(np.ndarray[DTYPE_t, ndim=2] a,
              np.ndarray[DTYPE_t, ndim=2] b):
    
    cdef int aN = a.shape[0]
    cdef int bN = b.shape[0]
    
    cdef int M = a.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=4] temp = np.zeros([aN, bN, 2, 2], dtype=DTYPE)
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros([aN, bN], dtype=np.float64)

    cdef int ai
    cdef int bi
    cdef int i

    cdef int ax
    cdef int bx

    cdef int A
    cdef int B
    cdef int AB

    cdef float ABp
    cdef float Ap
    cdef float Bp
    cdef float Ep
    cdef float d
    
    for ai in range(aN):
        for bi in range(bN):

            for i in range(M):
                
                ax = a[ai,i]
                bx = b[bi,i]
                
                temp[ai, bi, ax, bx] += 1

            AB = temp[ai, bi, 1, 1]
            A = (temp[ai, bi, 1, 1] +
                 temp[ai, bi, 1, 0])
            B = (temp[ai, bi, 1, 1] +
                 temp[ai, bi, 0, 1])

            if (A == 0) or (B == 0):
                result[ai, bi] = np.NaN

            else:
                ABp = AB / <float>M
                Ap = A / <float>M
                Bp = B / <float>M
                Ep = Ap * Bp

                d = ABp - Ep

                if d > 0.:

                    if (Ap <= Bp):

                        result[ai, bi] = d / (Ap - Ep)

                    elif (Bp <= Ap):

                        result[ai, bi] = d / (Bp - Ep)

                elif d < 0.:

                    result[ai, bi] = d / Ep

                else:

                    result[ai, bi] = 0.

    return result