# ============================================================
# NUMPY MAIN CONCEPTS
# ============================================================
# NumPy (Numerical Python) is the foundation of scientific computing in Python.
# Its core object is the ndarray: a fast, fixed-type, N-dimensional array.

import numpy as np

# ============================================================
# 1. CREATING ARRAYS
# ============================================================

a = np.array([1, 2, 3, 4, 5])                  # 1D from list
b = np.array([[1, 2, 3], [4, 5, 6]])            # 2D from nested list

print("--- Creating Arrays ---")
print(np.zeros((3, 4)))                          # All zeros, shape 3x4
print(np.ones((2, 3)))                           # All ones
print(np.full((2, 2), 7))                        # Filled with a constant
print(np.eye(3))                                 # Identity matrix
print(np.arange(0, 10, 2))                       # Like range(): [0 2 4 6 8]
print(np.linspace(0, 1, 5))                      # 5 evenly spaced values in [0,1]
print(np.random.rand(3, 3))                      # Uniform random floats [0,1)
print(np.random.randn(3, 3))                     # Standard normal distribution
print(np.random.randint(0, 10, size=(3, 3)))     # Random integers


# ============================================================
# 2. ARRAY ATTRIBUTES
# ============================================================

print("\n--- Array Attributes ---")
a2d = np.array([[1, 2, 3], [4, 5, 6]])
print("shape  :", a2d.shape)    # (2, 3) — rows, columns
print("ndim   :", a2d.ndim)     # 2 — number of dimensions
print("size   :", a2d.size)     # 6 — total number of elements
print("dtype  :", a2d.dtype)    # int64 — element data type
print("itemsize:", a2d.itemsize) # bytes per element


# ============================================================
# 3. DATA TYPES
# ============================================================
# NumPy arrays are typed — all elements share the same dtype.

print("\n--- Data Types ---")
print(np.array([1, 2, 3], dtype=np.float32))    # Specify dtype on creation
print(np.array([1.5, 2.7]).astype(np.int32))    # Cast to a different type

# Common dtypes:
# np.int8/16/32/64   np.uint8/16/32/64
# np.float16/32/64   np.complex64/128
# np.bool_           np.str_


# ============================================================
# 4. INDEXING & SLICING
# ============================================================

print("\n--- Indexing & Slicing ---")
a = np.array([10, 20, 30, 40, 50])
print(a[1])          # 20
print(a[-1])         # 50 (last element)
print(a[1:4])        # [20 30 40]
print(a[::2])        # [10 30 50] (every other)

b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(b[0, 1])       # 2  — row 0, col 1
print(b[:, 1])       # [2 5 8] — all rows, col 1
print(b[1:, :2])     # rows 1+, first 2 cols


# ============================================================
# 5. BOOLEAN INDEXING (FANCY INDEXING)
# ============================================================

print("\n--- Boolean Indexing ---")
a = np.array([10, 20, 30, 40, 50])
mask = a > 25
print(mask)          # [False False  True  True  True]
print(a[mask])       # [30 40 50]
print(a[a % 20 == 0]) # [20 40] — elements divisible by 20

# Integer array indexing
idx = np.array([0, 2, 4])
print(a[idx])        # [10 30 50]


# ============================================================
# 6. RESHAPING & TRANSPOSING
# ============================================================

print("\n--- Reshaping ---")
a = np.arange(12)
print(a.reshape(3, 4))          # 1D -> 2D (3 rows, 4 cols)
print(a.reshape(2, 2, 3))       # 1D -> 3D
print(a.reshape(3, -1))         # -1 lets NumPy infer the size

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.T)                       # Transpose (swap rows and columns)
print(b.flatten())               # Copy as 1D array
print(b.ravel())                 # View as 1D (no copy if possible)


# ============================================================
# 7. ARITHMETIC & UNIVERSAL FUNCTIONS (ufuncs)
# ============================================================
# Operations apply element-wise by default.

print("\n--- Arithmetic ---")
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(a + b)          # [11 22 33 44]
print(b - a)          # [ 9 18 27 36]
print(a * b)          # [ 10  40  90 160]
print(b / a)          # [10. 10. 10. 10.]
print(a ** 2)         # [ 1  4  9 16]
print(b % 3)          # [1 2 0 1]

print(np.sqrt(a))
print(np.exp(a))
print(np.log(b))
print(np.abs(np.array([-1, -2, 3])))
print(np.sin(np.pi / 2))


# ============================================================
# 8. BROADCASTING
# ============================================================
# NumPy automatically expands smaller arrays to match shapes
# without copying data.

print("\n--- Broadcasting ---")
a = np.array([[1, 2, 3],
              [4, 5, 6]])       # shape (2, 3)
b = np.array([10, 20, 30])     # shape (3,) -> broadcast to (2, 3)
print(a + b)
# [[11 22 33]
#  [14 25 36]]

scalar = 100
print(a + scalar)               # scalar broadcasts to every element

col = np.array([[1], [2]])      # shape (2, 1) -> broadcast to (2, 3)
print(a * col)


# ============================================================
# 9. AGGREGATION & STATISTICS
# ============================================================

print("\n--- Aggregation ---")
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.sum())               # 21  — total
print(a.sum(axis=0))         # [5 7 9]  — column sums
print(a.sum(axis=1))         # [ 6 15]  — row sums
print(a.min(), a.max())
print(a.mean(), a.std(), a.var())
print(a.argmin(), a.argmax()) # index of min/max
print(np.median(a))
print(np.percentile(a, 75))
print(np.cumsum(a))           # cumulative sum


# ============================================================
# 10. SORTING
# ============================================================

print("\n--- Sorting ---")
a = np.array([3, 1, 4, 1, 5, 9, 2])
print(np.sort(a))             # sorted copy
print(np.argsort(a))          # indices that would sort the array
a.sort()                      # sort in place
print(a)

b = np.array([[3, 1], [2, 4]])
print(np.sort(b, axis=0))     # sort each column
print(np.sort(b, axis=1))     # sort each row


# ============================================================
# 11. STACKING & SPLITTING
# ============================================================

print("\n--- Stacking ---")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.concatenate([a, b]))            # [1 2 3 4 5 6]
print(np.vstack([a, b]))                 # stack rows vertically
print(np.hstack([a.reshape(-1,1),
                 b.reshape(-1,1)]))      # stack columns horizontally
print(np.stack([a, b], axis=0))          # new axis stacking

print("\n--- Splitting ---")
a = np.arange(9)
print(np.split(a, 3))                    # split into 3 equal parts
print(np.array_split(a, 4))             # unequal splits allowed

b = np.arange(16).reshape(4, 4)
print(np.vsplit(b, 2))                   # split into top/bottom halves
print(np.hsplit(b, 2))                   # split into left/right halves


# ============================================================
# 12. LINEAR ALGEBRA
# ============================================================

print("\n--- Linear Algebra ---")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A @ B)                     # Matrix multiplication (also np.matmul)
print(np.dot(A, B))              # Dot product (same for 2D)
print(np.linalg.det(A))          # Determinant
print(np.linalg.inv(A))          # Inverse
print(np.linalg.eig(A))          # Eigenvalues and eigenvectors
print(np.linalg.norm(A))         # Matrix norm
U, S, Vt = np.linalg.svd(A)     # Singular Value Decomposition
solution = np.linalg.solve(A, np.array([1, 2]))  # Solve Ax = b


# ============================================================
# 13. RANDOM NUMBER GENERATION
# ============================================================

print("\n--- Random ---")
rng = np.random.default_rng(seed=42)   # Recommended modern API

print(rng.random((2, 3)))              # Uniform [0, 1)
print(rng.integers(0, 10, size=(3, 3))) # Random ints
print(rng.normal(loc=0, scale=1, size=5))  # Normal distribution
print(rng.choice([10, 20, 30, 40], size=3, replace=False))  # Sample without replacement

arr = np.arange(10)
rng.shuffle(arr)                       # Shuffle in place
print(arr)


# ============================================================
# 14. COPYING vs VIEWS
# ============================================================
# Slices return a VIEW (no copy) — modifying it modifies the original.
# Use .copy() to make an independent copy.

print("\n--- Copy vs View ---")
a = np.array([1, 2, 3, 4, 5])

view = a[1:3]
view[0] = 99
print(a)             # [1 99 3 4 5] — original changed!

copy = a[1:3].copy()
copy[0] = 0
print(a)             # unchanged — copy is independent


# ============================================================
# 15. WORKING WITH NAN
# ============================================================

print("\n--- NaN ---")
a = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

print(np.isnan(a))                # [False  True False  True False]
print(a[~np.isnan(a)])            # filter out NaNs: [1. 3. 5.]
print(np.nansum(a))               # 9.0  — ignores NaN
print(np.nanmean(a))              # 3.0
print(np.nanmax(a))               # 5.0


# ============================================================
# 16. REINDEXING / ADVANCED INDEXING
# ============================================================
# NumPy doesn't have labeled indexes like pandas, but you can
# rearrange, select, and map data using integer and boolean arrays.

print("\n--- Reindexing / Advanced Indexing ---")
a = np.array([10, 20, 30, 40, 50])

# Reorder with an index array
order = np.array([2, 0, 4, 1, 3])
print(a[order])                   # [30 10 50 20 40]

# np.take — equivalent to fancy indexing, works on any axis
print(np.take(a, [0, 2, 4]))      # [10 30 50]

b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(np.take(b, [0, 2], axis=0)) # rows 0 and 2

# np.put — scatter values into specific positions (in-place)
a2 = np.zeros(5, dtype=int)
np.put(a2, [1, 3], [99, 88])
print(a2)                          # [ 0 99  0 88  0]

# np.ix_ — open mesh for selecting subsets of a 2D array
rows = np.array([0, 2])
cols = np.array([1, 2])
print(b[np.ix_(rows, cols)])       # [[2 3], [8 9]]

# Mapping values via an index table (lookup table pattern)
lut = np.array([100, 200, 300, 400])  # lookup table
indices = np.array([3, 0, 2, 1, 0])
print(lut[indices])                # [400 100 300 200 100]


# ============================================================
# QUICK REFERENCE CHEAT SHEET
# ============================================================
#
# CREATING
#   np.array([...])                      From list
#   np.zeros/ones/full(shape)            Constant arrays
#   np.arange(start, stop, step)         Range of values
#   np.linspace(start, stop, n)          Evenly spaced values
#   np.eye(n)                            Identity matrix
#   np.random.default_rng(seed)          Random generator
#
# ATTRIBUTES
#   arr.shape / arr.ndim / arr.size      Dimensions
#   arr.dtype / arr.itemsize             Type info
#
# INDEXING
#   arr[i] / arr[i, j]                  Element access
#   arr[start:stop:step]                 Slicing
#   arr[bool_mask]                       Boolean indexing
#   arr[int_array]                       Fancy indexing
#
# RESHAPING
#   arr.reshape(shape)                   Change shape
#   arr.T                                Transpose
#   arr.flatten() / arr.ravel()          To 1D
#
# MATH
#   +, -, *, /, **, %                    Element-wise ops
#   np.sqrt/exp/log/abs/sin/cos          Ufuncs
#   arr @ other / np.dot(a, b)           Matrix multiply
#
# AGGREGATION
#   arr.sum/min/max/mean/std(axis=)      Reductions
#   np.argmin/argmax/cumsum              Index & cumulative
#   np.nansum/nanmean/nanmax             NaN-safe versions
#
# COMBINING
#   np.concatenate([a, b])               Join along axis
#   np.vstack/hstack/stack               Stack arrays
#   np.split/vsplit/hsplit               Split arrays
#
# LINEAR ALGEBRA
#   np.linalg.det/inv/eig/svd/solve      Matrix operations
#
# REINDEXING
#   arr[index_array]                     Reorder via integers
#   np.take(arr, indices, axis=)         Select along an axis
#   np.put(arr, indices, values)         Scatter values in place
#   np.ix_(rows, cols)                   2D subset selection
#
# MISC
#   arr.copy()                           Independent copy (not a view)
#   np.isnan(arr) / ~np.isnan(arr)       Detect/filter NaN
#   arr.astype(dtype)                    Cast type
