import numpy as np



class TriangularLattice:
    ''' Triangular Lattice Class

    '''
    def __init__(self, ysize, xsize, init_perturb=0.01, rho0=100, tau=0.6):
        # create lattice array and fill in with empty nodes
        self.ysize = ysize
        self.xsize = xsize
        self.rho0 = rho0
        self.tau = tau
        # calculate the points for all the nodes
        self.ylocs, self.xlocs = np.meshgrid(range(self.ysize), range(self.xsize))
        offsets = np.array([0,0.5]*(self.ysize//2)+[0]*(self.ysize%2)).reshape(self.ysize,1)
        self.xlocs += offsets
        # set velocity weights
        self.xvel_weights = np.array([0,1,0.5,-0.5,-1,-0.5,0.5])
        self.yvel_weights = np.array([0,0,np.sqrt(3)/2,np.sqrt(3)/2,0,-np.sqrt(3)/2,-np.sqrt(3)/2])
        # setup initial conditions
        self.F = np.ones((ysize, xsize, 7)) + init_perturb*np.random.randn(ysize, xsize, 7)
        self.F[:,:,1] += 2*(1+0.2*np.cos(2*np.pi*np.arange(xsize)/xsize))
        # correct the density
        rho = np.sum(self.F, axis=2).reshape(ysize, xsize, 1)
        self.F *= self.rho0/rho
        # setup boundaries
        self.cylinder = (self.xlocs-self.xsize/4)**2 + (self.y-self.ysize/2)**2 < (self.ysize/4)**2

    def apply_drift(self):
        ''' Applies drift to the components of the lattice. '''
        newF = np.zeros_like(self.F)
        d_x_eveny = np.array([0,1,0,-1,-1,-1,0])
        d_x_oddy =  np.array([0,1,1,0,-1,0,1])
        d_y = np.array([0,0,-1,-1,0,1,1])
        xidxs = np.arange(self.xsize)

        for y in range(0, self.ysize, 2):
            # even rows!
            for i, dy, dx in zip(range(7), d_y, d_x_eveny):
                newF[y+dy, xidxs+dx, i] = self.F[y, xidxs, i]
        
        for y in range(1,self.ysize,2):
            # odd rows!
            for i, dy, dx in zip(range(7), d_y, d_x_oddy):
                newF[y+dy, xidxs+dx, i] = self.F[y, xidxs, i]

    def update_frame(self):
        ''' Updates the frame. '''
        # apply drift to all velocities
        self.apply_drift()

        # apply bounce back to the cylinder
        reflected_vels = [0,4,5,6,1,2,3]
        self.F[self.cylinder,:] = self.F[self.cylinder,reflected_vels]
        # apply bounce back to the top and bottom
        self.F[0,:] = self.F[0,reflected_vels]
        self.F[-1,:] = self.F[-1,reflected_vels]
        # calculate fluid variables
        rho = np.sum(self.F, axis=2)
        ux = np.sum(self.F * self.xvel_weights, axis=2)/rho
        uy = np.sum(self.F * self.yvel_weights, axis=2)/rho
        # calculate eq
        Feq = np.zeros_like(self.F)
        


    def get_pressure_frame(self):
        ''' Get a frame of the pressure values across the lattice.
        '''


        