import numpy as np
from skimage.draw import disk, line
import matplotlib.pyplot as plt
import os

class PhantomGenerator:
    def __init__(self, E, r, grid_size=128):
        self.E = E
        self.r = r
        self.grid_size = grid_size
        self.x = np.linspace(-r, r, grid_size)
        self.y = np.linspace(-r, r, grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        angles = np.linspace(0, 2 * np.pi, E, endpoint=False)
        self.electrodes = [(r * np.cos(a), r * np.sin(a)) for a in angles]

    def generate_phantom(self, objects):
        phantom = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obj in objects:
            cx, cy = obj['center']
            rr, cc = disk(((cy + self.r) / (2 * self.r) * (self.grid_size - 1),
                           (cx + self.r) / (2 * self.r) * (self.grid_size - 1)),
                          radius=obj['radius'] / (2 * self.r) * (self.grid_size - 1))
            phantom[rr, cc] = 1.0
        return phantom

    def compute_measurements(self, phantom):
        M = []
        E = self.E
        coords = []
        for ex, ey in self.electrodes:
            ix = int((ex + self.r) / (2 * self.r) * (self.grid_size - 1))
            iy = int((ey + self.r) / (2 * self.r) * (self.grid_size - 1))
            coords.append((iy, ix))

        for i in range(E):
            for j in range(i + 1, E):
                p1 = coords[i]
                p2 = coords[j]
                rr, cc = line(p1[0], p1[1], p2[0], p2[1])
                valid = ((rr >= 0) & (rr < self.grid_size) & (cc >= 0) & (cc < self.grid_size))
                vals = phantom[rr[valid], cc[valid]]
                if len(vals) > 0:
                    M.append(vals.mean())
                else:
                    M.append(0.0)
        return np.array(M)

def generatesamples(num_samples):
    samples = []
    for _ in range(num_samples):
        num_objects = np.random.randint(1, 5)  # Random number of objects per phantom
        objects = []
        for _ in range(num_objects):
            obj = {
                'center': (np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8)),
                'radius': np.random.uniform(0.05, 0.2)
            }
            objects.append(obj)
        samples.append(objects)
    return samples


if __name__ == "__main__":
    E = 12  # Number of electrodes
    r = 1.0
    grid_size = 128
    num_samples = 10000  # Number of samples to generate

    gen = PhantomGenerator(E, r, grid_size)
    samples = generatesamples(num_samples)

    combined_data = []
    for i, objects in enumerate(samples):
        print(i)
        phantom = gen.generate_phantom(objects)
        measurements = gen.compute_measurements(phantom)
        combined_data.append({'objects': objects, 'measurements': measurements})

        # Optionally, visualize the phantom
        #plt.figure(figsize=(6, 6))
        #plt.imshow(phantom, extent=[-r, r, -r, r], cmap='gray')
        #plt.scatter([e[0] for e in gen.electrodes], [e[1] for e in gen.electrodes],
        #            c='r', s=30)
        #plt.title(f'Phantom {i + 1}')
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.legend()
        #plt.axis('equal')
        #plt.show()

    output_dir = r"C:\Users\welov\PycharmProjects\ECHO_ML\DATA\GrantGeneratedData"
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "combined_data.npy")
        np.save(output_file, combined_data, allow_pickle=True)
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

    loaded_data = np.load(output_file, allow_pickle=True)
    print("Loaded data example:", loaded_data[0])