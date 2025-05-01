import numpy as np
from skimage.draw import disk, line
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import json
from skimage.draw import polygon



# Class to generate phantom images and compute measurements for ECT simulations
class PhantomGenerator:
    def __init__(self, E, r, grid_size=128):
        """
            Initialize the PhantomGenerator class.

            Parameters:
            - E: Number of electrodes
            - r: Radius of the circular domain
            - grid_size: Size of the grid (square image dimensions)
        """
        self.E = E  # Number of electrodes
        self.r = r  # Radius of the circular domain
        self.grid_size = grid_size  # Size of the grid (image resolution)
        self.x = np.linspace(-r, r, grid_size)  # Generate x-coordinates for the grid
        self.y = np.linspace(-r, r, grid_size)  # Generate y-coordinates for the grid
        self.X, self.Y = np.meshgrid(self.x, self.y)  # Create a 2D grid of coordinates
        angles = np.linspace(0, 2 * np.pi, E, endpoint=False)  # Divide the circle into equal angles for electrodes
        self.electrodes = [(r * np.cos(a), r * np.sin(a)) for a in angles]  # Compute electrode positions on the circle

    def animate_phantom(self, phantom, emitting_electrode_index=0):
        """
        Animate the phantom image with lines originating from the midpoint of the emitting arc
        to the midpoint of the receiving arc sequentially.

        Parameters:
        - phantom: 2D numpy array representing the phantom image
        - emitting_electrode_index: Index of the fixed emitting electrode
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(phantom, cmap="gray", extent=(-self.r, self.r, -self.r, self.r))
        ax.set_title("Phantom with Arc Midpoint Lines")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Plot all electrodes as red dots initially
        for ex, ey in self.electrodes:
            ax.plot(ex, ey, 'ro', markersize=5)  # Red dot for each electrode

        # Prepare the lines for animation
        lines = []
        angles = np.linspace(0, 2 * np.pi, self.E, endpoint=False)

        # Calculate the midpoint of the emitting arc
        theta1 = angles[emitting_electrode_index]
        theta2 = angles[(emitting_electrode_index + 1) % self.E]
        emit_arc_mid_angle = (theta1 + theta2) / 2
        emit_arc_mid_x = self.r * np.cos(emit_arc_mid_angle)
        emit_arc_mid_y = self.r * np.sin(emit_arc_mid_angle)

        for j in range(self.E):  # Loop through each receiving electrode
            if emitting_electrode_index != j:  # Skip self-connections
                # Calculate the midpoint of the receiving arc
                theta1 = angles[j]
                theta2 = angles[(j + 1) % self.E]
                recv_arc_mid_angle = (theta1 + theta2) / 2
                recv_arc_mid_x = self.r * np.cos(recv_arc_mid_angle)
                recv_arc_mid_y = self.r * np.sin(recv_arc_mid_angle)

                # Store the line coordinates
                lines.append((emit_arc_mid_x, emit_arc_mid_y, recv_arc_mid_x, recv_arc_mid_y))

        # Initialize a line object for the animation
        line, = ax.plot([], [], 'b-', linewidth=2)  # Blue line for the connection

        def update(frame):
            emit_x, emit_y, recv_x, recv_y = lines[frame]
            # Update the line to connect the emitting and receiving arc midpoints
            line.set_data([emit_x, recv_x], [emit_y, recv_y])
            return line,

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(lines), interval=500, repeat=False)

        plt.show()
    def plot_phantom_with_electrodes(self, phantom):
        """
        Plot the phantom image with electrodes represented as arcs.

        Parameters:
        - phantom: 2D numpy array representing the phantom image
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(phantom, cmap="gray", extent=(-self.r, self.r, -self.r, self.r))
        ax.set_title("Phantom with Electrodes and Arcs")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Plot arcs as electrodes
        for i in range(len(self.electrodes)):
            # Get the current and next electrode positions
            ex, ey = self.electrodes[i]
            next_ex, next_ey = self.electrodes[(i + 1) % len(self.electrodes)]  # Wrap around to the first electrode

            # Draw arcs between the current electrode and the next
            arc = patches.Arc((0, 0), 2 * self.r, 2 * self.r,
                              theta1=np.degrees(np.arctan2(ey, ex)),
                              theta2=np.degrees(np.arctan2(next_ey, next_ex)),
                              color="green" if i == 0 else "blue", linewidth=2)
            ax.add_patch(arc)

            # Add a dot or divider at the electrode position
            ax.plot(ex, ey, 'ro', markersize=5)  # Red dot for each electrode

        plt.grid(True)
        plt.show()

    def generate_phantom(self, objects):
        """
        Generate a phantom image with circular objects.

        Parameters:
        - objects: List of dictionaries, each containing the center and radius of a circular object

        Returns:
        - phantom: 2D numpy array representing the phantom image
        """
        # Initialize a blank grid (phantom) with all zeros
        phantom = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Loop through each object to add it to the phantom
        for obj in objects:
            cx, cy = obj['center']  # Extract the center coordinates of the object
            # Compute the pixel coordinates of the circle using the disk function
            rr, cc = disk(((cy + self.r) / (2 * self.r) * (self.grid_size - 1),
                           (cx + self.r) / (2 * self.r) * (self.grid_size - 1)),
                          radius=obj['radius'] / (2 * self.r) * (self.grid_size - 1))
            phantom[rr, cc] = 1.0  # Set the pixels within the circle to 1 (indicating the object)
        return phantom  # Return the generated phantom image

    def compute_measurements(self, phantom, emitting_electrode_index):
        """
        Compute capacitance measurements as the percentage of the area between two arcs
        covered by the phantom for a single emitting electrode.

        Parameters:
        - phantom: 2D numpy array of the phantom image
        - emitting_electrode_index: Index of the emitting electrode

        Returns:
        - M: 1D numpy array of capacitance measurements
        """



        M = []  # Initialize a list to store capacitance measurements
        E = self.E  # Number of electrodes
        angles = np.linspace(0, 2 * np.pi, E, endpoint=False)  # Angular positions of electrodes
        arc_points = 5000  # Number of points to sample along each arc

        # Define the emitting electrode endpoints
        theta1_emit = angles[emitting_electrode_index]
        theta2_emit = angles[(emitting_electrode_index + 1) % E]



        for j in range(E):  # Loop through each receiving electrode
            if emitting_electrode_index != j:  # Skip self-connections
                # Define the receiving electrode endpoints
                theta1_recv = angles[j]
                theta2_recv = angles[(j + 1) % E]

                # Generate points for the two arcs from the emitting electrode
                arc1_angles = np.linspace(theta1_emit, theta1_recv, arc_points)
                arc2_angles = np.linspace(theta2_emit, theta2_recv, arc_points)



                # Generate points for the two arcs
                arc1_xs = self.r * np.cos(arc1_angles)
                arc1_ys = self.r * np.sin(arc1_angles)
                arc2_xs = self.r * np.cos(arc2_angles)
                arc2_ys = self.r * np.sin(arc2_angles)

                # Combine the arcs to form the region between them
                xs = np.concatenate([arc1_xs, arc2_xs[::-1]])
                ys = np.concatenate([arc1_ys, arc2_ys[::-1]])

                # Convert real-world coordinates to grid indices
                ix = ((xs + self.r) / (2 * self.r) * (self.grid_size - 1)).astype(int)
                iy = ((ys + self.r) / (2 * self.r) * (self.grid_size - 1)).astype(int)

                # Ensure indices are within bounds
                valid = (ix >= 0) & (ix < self.grid_size) & (iy >= 0) & (iy < self.grid_size)
                ix, iy = ix[valid], iy[valid]

                # Create a mask for the region between the arcs
                rr, cc = polygon(iy, ix, phantom.shape)
                region_mask = np.zeros_like(phantom, dtype=bool)
                region_mask[rr, cc] = True

                # Create a mask for the pie slice
                rr, cc = polygon(iy, ix, phantom.shape)  # Fill the pie slice
                region_mask = np.zeros_like(phantom, dtype=bool)
                region_mask[rr, cc] = True

                # Debugging visualization
                plt.imshow(region_mask, cmap="gray")
                plt.title("Region Mask")
                plt.show()
                plt.imshow(phantom, cmap="gray")
                plt.title("Phantom")
                plt.show()


                # Calculate the percentage of the region covered by the phantom
                total_area = np.sum(region_mask)
                covered_area = np.sum(phantom[region_mask])
                coverage = covered_area / total_area if total_area > 0 else 0.0

                print(f"Total Area: {total_area}, Covered Area: {covered_area}")

                M.append(coverage)  # Append the coverage percentage

        return np.array(M)  # Convert the list of measurements to a numpy array


def generatesamples(num_samples):
    """
    Generate a set of random phantom samples.

    Parameters:
    - num_samples: Number of samples to generate

    Returns:
    - samples: List of phantom objects, each containing circular objects
    """
    samples = []  # Initialize an empty list to store samples
    for _ in range(num_samples):  # Loop to generate the specified number of samples
        num_objects = np.random.randint(1, 5)  # Randomly choose the number of objects in the phantom
        objects = []  # Initialize an empty list to store the objects
        for _ in range(num_objects):  # Loop to generate the specified number of objects
            obj = {
                'center': (np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8)),
                # Random center within the domain
                'radius': np.random.uniform(0.05, 0.2)  # Random radius within a specified range
            }
            objects.append(obj)  # Append the object to the list
        samples.append(objects)  # Append the list of objects to the samples
    return samples  # Return the generated samples


if __name__ == "__main__":
    # Configuration parameters
    ELECTRODE_NUM = 15 # Number of electrodes
    r = 1.0 # Radius of the circular domain
    grid_size = 128 # Size of the grid (image resolution)
    num_samples = 100 # Number of phantom samples to generate
    emitting_electrode = 0  # Index of emitting electrode

    # Initialize the PhantomGenerator
    gen = PhantomGenerator(ELECTRODE_NUM, r, grid_size)
    # Generate random phantom samples
    samples = generatesamples(num_samples)

    combined_data = []  # Initialize a list to store the combined data
    for i, objects in enumerate(samples):  # Loop through each sample
        print(i)  # Print the sample index
        phantom = gen.generate_phantom(objects)  # Generate the phantom image
        measurements = gen.compute_measurements(phantom,emitting_electrode)  # Compute the capacitance measurements
        combined_data.append({'objects': objects, 'measurements': measurements})  # Store the objects and measurements

    gen.plot_phantom_with_electrodes(phantom)
    gen.animate_phantom(phantom)  # Animate the phantom with electrodes

    # Save the generated data to a file
    output_dir = r"C:\Users\welov\PycharmProjects\ECHO_ML\DATA\GrantGeneratedData"  # Absolute path to the desired directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "test5_1emit.json")

        # Convert `ndarray` to list for JSON serialization
        for data in combined_data:
            data['measurements'] = data['measurements'].tolist()

        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(combined_data, f)
        print(f"Data successfully saved to {output_file}")

        # Load the JSON file
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)

        # Convert lists back to `ndarray` if needed
        for data in loaded_data:
            data['measurements'] = np.array(data['measurements'])

        print("Loaded data example:", loaded_data[0])
    except Exception as e:
        print(f"Error saving file: {e}")