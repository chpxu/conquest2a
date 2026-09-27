.. _types:

Types
=====

C2a defines custom types, mostly wrapping around NumPy types to leverage full static typing:

* ``GENERIC_ARRAY = npt.NDArray[typing.Any]`` - corresponds to an array of anything
* ``REAL_ARRAY = npt.NDArray[np.float64]`` - corresponds to an array of floats
* ``INT_ARRAY = npt.NDArray[np.int64]`` - corresponds to an array of integers

