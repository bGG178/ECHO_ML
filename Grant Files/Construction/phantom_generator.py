import numpy as np
from skimage.draw import disk, line
import matplotlib.pyplot as plt
import os

class PhantomGenerator:
    def __init__(self, E, r, grid_size=128): # E = number of electrodes, r = radius of object?
        self.E = E
        self.r = r
        self.grid_size = grid_size
        self.x = np.linspace(-r, r, grid_size) # x-grid
        self.y = np.linspace(-r, r, grid_size) # y-grid
        self.X, self.Y = np.meshgrid(self.x, self.y) # create the whole grid
        angles = np.linspace(0, 2 * np.pi, E, endpoint=False) #angles for electrode placement
        self.electrodes = [(r * np.cos(a), r * np.sin(a)) for a in angles] # create the electrode layout in angle form

    def generate_phantom(self, objects): # create the actual phantom, the objects are defined later
        phantom = np.zeros((self.grid_size, self.grid_size), dtype=np.float32) # create a blank background?
        for obj in objects: # where is "objects" defined, is it the number of objects?
            cx, cy = obj['center'] # find the center of the grid? or the center of the object?
            rr, cc = disk(((cy + self.r) / (2 * self.r) * (self.grid_size - 1), # create a circle on the grid?
                           (cx + self.r) / (2 * self.r) * (self.grid_size - 1)),
                          radius=obj['radius'] / (2 * self.r) * (self.grid_size - 1))
            phantom[rr, cc] = 1.0 # what does this mean?
        return phantom

    def compute_measurements(self, phantom): # scan the phantom to determine capacitance measurements
        M = [] # growing vector of capacitance values
        E = self.E # number of electrodes
        coords = [] # start a growing matrix of coordinates
        for ex, ey in self.electrodes: # for each electrodes x, y coordinate
            ix = int((ex + self.r) / (2 * self.r) * (self.grid_size - 1)) # electrode coordinates
            iy = int((ey + self.r) / (2 * self.r) * (self.grid_size - 1))
            coords.append((iy, ix)) # add these coordinates to the electrode coordinates

        for i in range(E): # for the number of electrodes, determining which electrode is scanning
            for j in range(i + 1, E): # for the number of electrodes that aren't emitting, does this loop back around
                p1 = coords[i] # coordinates of the receiving electrode
                p2 = coords[j]
                rr, cc = line(p1[0], p1[1], p2[0], p2[1]) # does this just create a line between each electrode? this should be an arc
                valid = ((rr >= 0) & (rr < self.grid_size) & (cc >= 0) & (cc < self.grid_size)) # what criteria is being used to determine if the measured values are valid?
                vals = phantom[rr[valid], cc[valid]] # measurement values of the phantom
                if len(vals) > 0: # check if any measurement values were collected
                    M.append(vals.mean()) # average the measurement values
                else:
                    M.append(0.0) # if no values were collected, state that (why not just only do the mean?)
        return np.array(M)

def generatesamples(num_samples):
    samples = [] # growing array of samples
    for _ in range(num_samples): # for the number of samples, generate samples
        num_objects = np.random.randint(1, 5)  # Random number of objects per phantom
        objects = [] # growing array of circles
        for _ in range(num_objects): # for the number of objects, generate circles in the phantom
            obj = {
                'center': (np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8)), # center of the circle
                'radius': np.random.uniform(0.05, 0.2) # radius of the circle
            }
            objects.append(obj) # add the details of the circle to the array of objects
        samples.append(objects) # add the objects created to the list of samples
    return samples


if __name__ == "__main__":
    ELECTRODE_NUM = 12  # Number of electrodes
    r = 1.0 # domain radius?
    grid_size = 128 # image size (square image0
    num_samples = 100  # Number of samples to generate

    gen = PhantomGenerator(ELECTRODE_NUM, r, grid_size) # generate phantoms
    samples = generatesamples(num_samples) # generate sample images

    combined_data = [] # create a growing array of measurement data
    for i, objects in enumerate(samples): # for the number of samples
        print(i) # print the sample
        phantom = gen.generate_phantom(objects) # generate phantoms from the samples
        measurements = gen.compute_measurements(phantom) # collect the measurement data form the samples
        combined_data.append({'objects': objects, 'measurements': measurements}) # add to the list of measurements

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

    output_dir = r"/DATA/GrantGeneratedData" # store the generated samples and data here
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "combined_datatest.npy")
        np.save(output_file, combined_data, allow_pickle=True)
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

    loaded_data = np.load(output_file, allow_pickle=True)
    print("Loaded data example:", loaded_data[0])