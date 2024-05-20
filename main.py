import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from mesh import generate_triangular_mesh, Point, plot_mesh, refine_mesh_near_point
from matplotlib.animation import FuncAnimation

def solve_heat_equation(triangulation, points):
    num_points = len(points)
    A = lil_matrix((num_points, num_points))
    b = np.zeros(num_points)

    for simplex in triangulation.simplices:
        for i in range(3):
            for j in range(3):
                A[simplex[i], simplex[j]] += 2 if i == j else -1

    # Introduce a heat source in the center of the domain
    center_index = num_points // 2  # Simplistic way to find a central point
    b[center_index] = 100  # Arbitrary heat source strength

    # Apply non-zero boundary conditions
    for i, point in enumerate(points):
        if point.x == 0 or point.x == 10:
            A[i, :] = 0
            A[i, i] = 1
            b[i] = 50  # Set left and right boundaries to 50
        elif point.y == 0 or point.y == 10:
            A[i, :] = 0
            A[i, i] = 1
            b[i] = 20  # Set top and bottom boundaries to 20

    temperatures = spsolve(A.tocsr(), b)

    # Compute average temperature for each triangle
    triangle_temperatures = np.array([np.mean(temperatures[simplex]) for simplex in triangulation.simplices])

    return triangle_temperatures

def plot_heat_distribution(triangulation, points, triangle_temperatures):
    plt.figure()
    plt.tripcolor(
        [p.x for p in points], 
        [p.y for p in points], 
        triangulation.simplices, 
        facecolors=triangle_temperatures, 
        cmap='hot', 
        edgecolors='k'
    )
    plt.colorbar()
    plt.scatter([p.x for p in points], [p.y for p in points], color='blue', s=10)  # Plot all nodes
    plt.show()

def animate_refinement(points, specified_points, radius, refinement_factor, min_distance, tries, shape, save_path=None):
    refined_points = points
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    scat = ax.scatter([p.x for p in points], [p.y for p in points], color='blue', s=10)
    lines = []

    def update(frame):
        nonlocal lines
        scat.set_offsets(np.array([[p.x, p.y] for p in frame]))
        
        # Remove previous lines if they exist
        while lines:
            line = lines.pop()
            line.remove()
        
        # Create new triangulation and plot edges
        points_array = np.array([[p.x, p.y] for p in frame])
        triangulation = Delaunay(points_array)
        lines = ax.triplot(points_array[:, 0], points_array[:, 1], triangulation.simplices, color='gray')
        
        return scat, *lines

    # Generate frames by refining the mesh step-by-step
    frames = []
    for specified_point in specified_points:
        for refined_points in refine_mesh_near_point(refined_points, specified_point, radius, refinement_factor, min_distance, tries, shape):
            frames.append(refined_points.copy())

    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=True, repeat=False)  # Increased interval to slow down

    # Uncomment to save the animation. Works only if ffmpeg is installed.
    # if save_path:
    #     ani.save(save_path, writer='ffmpeg', fps=60)

    plt.show()

    return frames[-1]  # Return the final refined points

def main():
    width, height = 10.0, 10.0
    num_points_x, num_points_y = 10, 10
    num_boundary_points_per_side = 10
    top_wall_temperature = 100

    points, triangulation = generate_triangular_mesh(
        width, height, num_points_x, num_points_y, num_boundary_points_per_side, top_wall_temperature
    )
    
    # Refine the mesh near specified points
    specified_points = [Point(5, 5)]  # Example specified points
    radius = 1.0  # Radius around the specified points to refine
    refinement_factor = 100  # Number of additional points to add around each point within the radius
    min_distance = 0.3  # Minimum distance between points
    tries = 30  # Number of tries to find a suitable point
    shape = 'circle'  # Shape of the refinement (circle or rectangle)
    
    # Animate the refinement process and get the final refined points
    refined_points = animate_refinement(points, specified_points, radius, refinement_factor, min_distance, tries, shape, save_path='refinement_animation.mp4')
    
    # Generate the triangulation with the refined points
    points_array = np.array([[p.x, p.y] for p in refined_points])
    refined_triangulation = Delaunay(points_array)
    
    # Solve the heat equation on the refined mesh
    triangle_temperatures = solve_heat_equation(refined_triangulation, refined_points)
    
    # Plot the heat distribution
    plot_heat_distribution(refined_triangulation, refined_points, triangle_temperatures)

if __name__ == "__main__":
    main()
