import numpy as np
import Trans3D

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
    [35, -35, 0],
    [-35, -35, 0]
])/1000.0

# Compute the affine transformation matrix
transformation_matrix = Trans3D.affine_matrix_from_points(points.T, target_points.T, False, False, True)

# Verify the transformation for original points
points_homogeneous = np.vstack((points.T, np.ones((1, points.shape[0]))))
transformed_points = np.dot(transformation_matrix, points_homogeneous)
transformed_points = transformed_points[:-1, :].T

# Check if the transformation is correct
is_correct = np.allclose(transformed_points, target_points, atol=1e-5)

# Print results for original points
print("Affine Transformation Matrix:\n", transformation_matrix)
print("Transformed Points:\n", transformed_points)
print("Target Points:\n", target_points)
print("Is the transformation correct?:", is_correct)

# Additional points to transform
coordinates = np.array([
    [0.0693237, 0.06697819, 0.2819021],
    [0.127036, 0.04104219, 0.31184563],
    [0.16635457, 0.08499814, 0.2741372],
    [0.10864227, 0.11093415, 0.24419369]
], dtype=np.float32)

# Transform the additional points using the same transformation matrix
coordinates_homogeneous = np.vstack((coordinates.T, np.ones((1, coordinates.shape[0]))))
transformed_coordinates = np.dot(transformation_matrix, coordinates_homogeneous)

# Extract the transformed coordinates
transformed_coordinates = transformed_coordinates[:-1, :].T

# Rescale or adjust the coordinates if needed (optional step)
# Depending on how well the transformed coordinates match the target, you might need additional scaling.
# Example: scaling to the range of the target points

# Output the transformed coordinates
print("Transformed Additional Points:\n", transformed_coordinates)

