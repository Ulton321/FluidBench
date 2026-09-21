import numpy as np
import matplotlib.pyplot as plt


def main(): # define our constants
    Nx = 400 # We want 400 cells in our X demension
    Ny = 100 # Y demension
    tau =  .53 #Our kinematic viscosity / Time scale
    Nt = 3000 #The amount of iteration
    rho0 = 100 # Average density

    # Lattices speed and weight:

    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, -1]) 
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
    X, Y = np.meshgrid(range(Nx), range(Ny))
    
    #intail the condition: 
    F = np.ones((Ny,Nx,NL)) + 0.01*np.random.randn(Ny,Nx,NL)
    F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
    rho = np.sum(F,2)
    for i in idxs:
        F[:,:,i] *= rho0 / rho

    # Cyliner Boundery

    cd = (X - Nx/4)**2 + (Y - Ny/2)**2 < (Ny/4)**4

   

    

if __name__ == "main":
     # Main loop
    
    for it in range(Nt):
            




    main()