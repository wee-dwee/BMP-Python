import numpy as np

# Provided coordinates
coordinates = np.array([
    [0.0693237, 0.06697819, 0.2819021],
    [0.127036, 0.04104219, 0.31184563],
    [0.16635457, 0.08499814, 0.2741372],
    [0.10864227, 0.11093415, 0.24419369]
], dtype=np.float32)

# Given transformation matrix
transformation_matrix = np.array([
   [-0.13029779 ,-0.46945037, -0.87329195 , 0.24846132],
 [-0.59605756 ,-0.66676865  ,0.44736445 ,-0.12125204],
 [-0.7922991  , 0.57882287 ,-0.19294096  ,0.08702267],
 [ 0.         , 0.        ,  0.,          1.        ]
])

# Step 1: Convert the coordinates to homogeneous form (add a column of ones)
coordinates_homogeneous = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))

# Step 2: Apply the transformation matrix
transformed_coordinates = np.dot(coordinates_homogeneous, transformation_matrix.T)

# Step 3: Extract the transformed points (ignore the homogeneous coordinate)
transformed_points = transformed_coordinates[:, :-1]

# Target points
target_points = np.array([
    [-35, 35, 0],
    [35, 35, 0],
    [-35, -35, 0],
    [35, -35, 0]
])

# Step 4: Calculate Euclidean distances between transformed points and target points
distances = np.linalg.norm(transformed_points - target_points, axis=1)

# Output the results
print("Transformed Points:\n", transformed_points)
print("Target Points:\n", target_points)
print("Distances:\n", distances)
