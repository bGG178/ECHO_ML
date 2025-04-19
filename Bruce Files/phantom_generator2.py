import numpy as np
from skimage.draw import disk, line
import matplotlib.pyplot as plt
import os
import math

# class that creates phantom images
class PhantomGenerator:
    def __init__(self, E, r, grid_size=128):
        self.E = E # number of electrodes
        self.r = r # radius of
        self.grid_size = grid_size #
        self.x = np.linspace(-r, r, grid_size) # x-grid
        self.y = np.linspace(-r, r, grid_size) # y-grid
        self.X, self.Y = np.meshgrid(self.x, self.y) # create base for image
        angles = np.linspace(0, 2 * np.pi, E, endpoint=False) # angles for electrode placement
        self.electrodes = [(r * np.cos(a), r * np.sin(a)) for a in angles] # electrode coordinates

    ## this needs to be changed to restrict disks to generate within capacitance ring
    def generate_phantom(self, objects): # phantom image
        phantom = np.zeros((self.grid_size, self.grid_size), dtype=np.float32) # create a blank background
        for obj in objects: # for each disk
            cx, cy = obj['center'] # center coordinates of each disk
            rr, cc = disk(((cy + self.r) / (2 * self.r) * (self.grid_size - 1),
                           (cx + self.r) / (2 * self.r) * (self.grid_size - 1)),
                          radius = obj['radius'] / (2 * self.r) * (self.grid_size - 1))
            phantom[rr, cc] = 1.0 # what does this mean?
        return phantom

    ## this needs to be changed to reflect soft field nature of ECT
    def compute_measurements(self, phantom): # scan the phantom to determine capacitance measurements
        M = [] # growing vector of capacitance measurements
        E = self.E # number of electrodes
        coords = [] # start a growing matrix of coordinates
        for ex, ey in self.electrodes: # for the xy coordinates of each electrode
            ix = int((ex + self.r) / (2 * self.r) * (self.grid_size - 1)) # electrode coordinates?
            iy = int((ey + self.r) / (2 * self.r) * (self.grid_size - 1))
            coords.append((iy, ix)) # add coordinate electrodes

        # for i in range(E): # cycle through emitting electrodes
            # for j in range(i + 1, E): # cycle through recieving electrodes
                # p1 = coords[i] # emitting electrode coordinates
                # p2 = coords[j] # recieving electrodes
                ## this needs to be changed to arcs
                # rr, cc = line(p1[0], p1[1], p2[0], p2[1])
                # xDif = p1[0]
                ## this is supposed to check if it is within the electrode ring, but it isn't working
                # valid = ((rr >= 0) & (rr < self.grid_size) & (cc >= 0) & (cc < self.grid_size)) # check that it is within the electrode ring
                # vals = phantom[rr[valid], cc[valid]] # measurement values from the phantom
                ## this check could probably just be removed
                # if len(vals) > 0: # check if any values were collected
                    ## averaging wrong, needs to be over length of the scan distance
                    # M.append(vals.mean()) # average the measured values and add them to the matrix
                # else:
                    # M.append(0.0) # add zero if no values
        # return np.array(M)

        ## scan with electrodes
        grid_size = self.grid_size
        r = self.r
        xc = grid_size/2
        yc = grid_size/2
        TK = 2*r/E # thickness of scan area
        center = grid_size/2 # center of image
        p1 = coords[1] # coordinates of first electrode
        DCMThetas = np.linspace(0,2*np.pi-2*np.pi/E,E)-np.pi/2
        for i in E:
            for ii in (E-1):
                cData = 0
                cCount = 0
                p2 = coords[ii+1]
                xDif= p1[0]-p2[0]
                if (xDif == 0):
                    ys = np.linspace(p1[1], p2[1], grid_size, endpoint=False) # consider changing how many points there are (grid_size)
                    xs = 0*ys
                else:
                    if (xDif > 0):
                        c = 1
                    else:
                        c = -1
                    # determine radius for required intercepts to be tangent
                    x = abs(p2[0]-p1[0])
                    y = abs(p2[1]-p1[0])
                    theta = math.atan(y/x)
                    phi = 2*theta-np.pi/2
                    rs = x+y*math.tan(phi)
                    # determine the xy points to plot
                    ## replace this with an angle sweep
                    xs = (rs)*math.cos(np.linspace(0,2*np.pi,1000))+c*rs
                    ys = (rs)*math.sin(np.linspace(0,2*np.pi,1000))+p1[1]
                DCM = -[[math.cos(DCMThetas[i]), -math.sin(DCMThetas[i])],...
                        [math.sin(DCMThetas[i]), math.cos(DCMThetas[i])]]
                arc = DCM*[[xs],[ys]]
                # shift coordinates relative to the center of the image
                arc[0] = arc[0]+xc
                arc[1] = arc[1]+yc
                # collect color data
                for iii in len(arc[0]):
                    if np.sqrt(((arc[0][iii] - xc) ^ 2 + (arc[2][iii] - yc) ^ 2) < ((r - TK / 2) - 1)):
                        xim = round(arc[0][iii], 1)
                        yim = round(arc[1][iii], 1)
                        xa = np.arange(int(round(xim - (TK/2)), int(round(xim + (TK/2))) + 1))
                        ya = np.arange(int(round(yim - (TK/2)), int(round(yim + (TK/2))) + 1))
                        cd = np.zeros((len(xa), len(ya)), dtype=np.float32)
                        # for each of the pixels being scanned, collect and average the data
                        for iiix in len(xa):
                            for iiiy in len(ya):
                                cd[iiix][iiiy] = im(xa[iiix],ya[iiiy],:)
                                cd[iiix][iiiy] = phantom[rr[valid], cc[valid]]



def generatesamples(num_samples):
    samples = [] # growing array of samples
    for _ in range(num_samples): # for the number of samples, generate samples
        num_objects = np.random.randint(1, 5) # random number of objects per phantom
        objects = [] # growing array of circles
        for _ in range(num_objects): # generate the objects in the phantom
            obj = {
                'center': (np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8)), # center of circle
                'radius': np.random.uniform(0.05, 0.2) # radius of the object
            }
            objects.append(obj) # add objects to the array of objects
        samples.append(objects) # add the objects to the list of samples
    return samples

if __name__ == "main":
    ELECTRODE_NUM = 12 # number of electrodes
    r = 1.0 # domain radius?
    grid_size = 128 # image size (square image)
    num_samples = 100 # number of samples generated

    gen = PhantomGenerator(ELECTRODE_NUM, r, grid_size) # generate phantoms
    samples = generate_samples(num_samples) # generate sample images

    combined_data = [] # growing array of measurement data
    for i, objects in enumerate(samples): # for the number of samples
        print(i) # display the sample
        phantom = gen.generate_phantom(objects) # generate phantoms from samples
        measurements = gen.compute_measurements(phantom) # collect measurement data
        combined_data.append({'objects': objects, 'measurements': measurements}) # add measurements to array

    output_dir = r"/DATA/GrantGeneratedData" # store generated samples and measurements
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "combined_datatest.npy")
        np.save(output_file, combined_data, allow_pickle=True)
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

    loaded_data = np.load(output_file, allow_pickle=True)
    print("Loaded data example:", loaded_data[0])