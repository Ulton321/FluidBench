import numpy as np
import matplotlib.pyplot as plt


def main(): # define our constants
    Nx = 400 # We want 400 cells in our X demension
    Ny = 100 # Y demension
    tau =  .53 #Our kinematic viscosity / Time scale
    Nt = 3000 #The amount of iteration

    # Lattices speed and weight:

    NL = 9
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, -1]) 
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    #intail the condition: 
    F = np.ones((Ny, Nx, NL)) + 0.01*np.random.randn(Nx, Ny, NL)









if __name__ == "main": 
    main()