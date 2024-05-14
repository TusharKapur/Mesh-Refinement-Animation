import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

class Point:
    def __init__(self, x, y, temperature=None):
        self.x = x
        self.y = y
        self.temperature = temperature

def generate_boundary_points(width, height, num_points_per_side, regular=True):
    boundary_points = []
    for i in range(num_points_per_side):
        fraction = i / (num_points_per_side - 1)
        boundary_points.extend([
            Point(fraction * width, 0),  # Bottom side
            Point(fraction * width, height),  # Top side
            Point(0, fraction * height),  # Left side
            Point(width, fraction * height)  # Right side
        ])
    return boundary_points

def generate_grid_points(width, height, num_points_x, num_points_y, regular=False):
    if regular:
        x_values = np.linspace(0, width, num_points_x)
        y_values = np.linspace(0, height, num_points_y)
        return [Point(x, y) for x in x_values for y in y_values]
    else:
        return [Point(np.random.uniform(0, width), np.random.uniform(0, height)) for _ in range(num_points_x * num_points_y)]

def plot_mesh(points, triangulation):
    plt.triplot([p.x for p in points], [p.y for p in points], triangulation.simplices)
    plt.scatter([p.x for p in points], [p.y for p in points], color='red', s=10)  # Plot the nodes
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Triangular Mesh')
    plt.show()

def generate_triangular_mesh(width, height, num_points_x, num_points_y, num_boundary_points_per_side, top_wall_temperature):
    # Generate grid points
    x = np.linspace(0, width, num_points_x)
    y = np.linspace(0, height, num_points_y)
    points = [Point(xi, yi) for xi in x for yi in y]

    # Set temperature for top wall points
    for p in points:
        if p.y == height:
            p.temperature = top_wall_temperature

    # Convert points to a format suitable for Delaunay triangulation
    points_array = np.array([[p.x, p.y] for p in points])
    triangulation = Delaunay(points_array)

    return points, triangulation

def smooth_points(points, min_distance):
    """
    Adjusts the positions of points to ensure a minimum distance between them.
    """
    points_array = np.array([[p.x, p.y] for p in points])
    for i, point in enumerate(points):
        for j, other_point in enumerate(points):
            if i != j:
                distance = np.linalg.norm(points_array[i] - points_array[j])
                if distance < min_distance:
                    # Move points away from each other
                    direction = (points_array[i] - points_array[j]) / distance
                    move_distance = (min_distance - distance) / 2
                    points_array[i] += direction * move_distance
                    points_array[j] -= direction * move_distance

    # Update the points with the new positions
    for i, point in enumerate(points):
        point.x, point.y = points_array[i]

    return points

def refine_mesh_near_point(points, specified_point, radius, refinement_factor, min_distance):
    refined_points = points.copy()
    for point in points:
        distance = np.sqrt((point.x - specified_point.x) ** 2 + (point.y - specified_point.y) ** 2)
        if distance < radius:
            # Add random points within the specified radius around the point
            for _ in range(refinement_factor):
                for _ in range(10):  # Try up to 10 times to find a suitable point
                    angle = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(0, radius)
                    new_x = point.x + r * np.cos(angle)
                    new_y = point.y + r * np.sin(angle)
                    new_point = Point(new_x, new_y)
                    
                    # Check if the new point is too close to existing points
                    if all(np.sqrt((new_point.x - p.x) ** 2 + (new_point.y - p.y) ** 2) >= min_distance for p in refined_points):
                        refined_points.append(new_point)
                        refined_points = smooth_points(refined_points, min_distance)  # Smooth the points
                        yield refined_points  # Yield the intermediate state
                        break

