import numpy as np

# Original Points
points = np.array([
    [-0.01920699, -0.025381, 0.28585267],
    [0.03558045, -0.05044783, 0.32148957],
    [0.07730448, -0.00377402, 0.29017407],
    [0.02251704, 0.0212928, 0.25453714]
])

# Desired Target Points
target_points = np.array([
    [-35, 35, 0],
    [35, 35, 0],
    [-35, -35, 0],
    [35, -35, 0]
])

def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)
    
    ndims = v0.shape[0]
    
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("Input arrays are of wrong shape or type")

    # Move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
    else:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R

    # Move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]  # Normalize the matrix
    return M

# Compute the affine transformation matrix
transformation_matrix = affine_matrix_from_points(points.T, target_points.T)

# Verify the transformation
# Transform original points using the transformation matrix
# Add a row of ones for homogeneous coordinates
points_homogeneous = np.vstack((points.T, np.ones((1, points.shape[0]))))
print(points_homogeneous)
transformed_points = np.dot(transformation_matrix, points_homogeneous)

# Extract the transformed points (ignore the last row)
transformed_points = transformed_points[:-1, :].T

# Compare transformed points with target points
is_correct = np.allclose(transformed_points, target_points, atol=1e-5)

# Output results
print("Affine Transformation Matrix:\n", transformation_matrix)
print("Transformed Points:\n", transformed_points)
print("Target Points:\n", target_points)
print("Is the transformation correct?:", is_correct)
