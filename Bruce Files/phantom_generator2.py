import numpy as np
from skimage.draw import disk, line
import matplotlib.pyplot as plt
import os

# class that creates phantom images
class PhantomGenerator:
    def __init__(selfself, E, r, grid_size=128):
        self.E = E
        self.r = r
        self.grid_size = grid_size
        self.x = np.linspace(-r, r, grid_size) # x-grid
        self.y = np.linspace(-r, r, grid_size) # y-grid
