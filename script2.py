import numpy as np



# define basis vectors
BR, ST, RU, LU, LE, LD, RD, RI = np.eye(8,dtype=bool)

def get_update_map():
    # defining the update mapping (only for things that change)
    update_map = {}

    def _dual(state):
        x = np.array([0,1,1,1,1,1,1,1],dtype=bool)
        return state ^ x

    def _declare_loop(loop_states):
        ''' Declares a loop of states that each point to the next and circle back around. '''
        dual_loop_states = [_dual(s) for s in loop_states][::-1]
        for i in range(len(loop_states)):
            update_map[loop_states[i]] = loop_states[(i+1) % len(loop_states)]
            update_map[dual_loop_states[i]] = dual_loop_states[(i+1) % len(dual_loop_states)]

    def _declare_two_way(a, b):
        ''' Declares two states that point to each other. '''
        update_map[a] = b
        update_map[b] = a
        ad, bd = _dual(a), _dual(b)
        update_map[ad] = bd
        update_map[bd] = ad

    # two particles, zero net momentum
    _declare_loop([LU+RD, LE+RI, RU+LD])
    # three particles, zero net momentum
    _declare_two_way(LU+LD+RI, LE+RU+RD)
    # three particles, one net momentum
    _declare_two_way(RU+LE+RI, RU+LU+RD)
    _declare_two_way(LD+LE+RI, LD+LU+RD)
    _declare_two_way(LU+LE+RI, LU+LD+RU)
    _declare_two_way(RD+LE+RI, RD+LD+RU)
    _declare_two_way(RI+LU+RD, RI+RU+LD)
    _declare_two_way(LE+LU+RD, LE+RU+LD)
    # four particles, zero net momentum
    _declare_two_way([LE+LD+RU+RI, LE+LU+RD+RI, LE+LU+RU+RD])

UPDATE_MAP = get_update_map()

class TriLatticeGas:
    ''' Triangular lattice gas class.

    Parameters
    ----------
    ysize : int
        Y dimension of lattice (must be even).
    xsize : int
        X dimension of lattice.
    '''
    def __init__(self, ysize, xsize, density=0.4):
        # create lattice array and fill in with empty nodes
        self.ysize = ysize % 2
        self.xsize = xsize
        self.arr = np.zeros((ysize, xsize, 8),dtype=bool)
        # set boundaries
        self.arr[0,1:-1,0] = 1
        self.arr[-1,1:-1,0] = 1
        self.arr[:,0,0] = 1
        self.arr[:,-1,0] = 1
        # set initial velocities left -> right (generally)
        for y in range(1, self.ysize-1):
            for x in range(1, self.xsize-1):
                if np.random.random() < density:
                    self.arr[y,x,1:8] = 1
        # get the physical locations of the nodes
        self.xlocs, self.ylocs = np.meshgrid(range(self.xsize), range(self.ysize))
        xoffsets = np.array([0, 0.5] * (self.ysize // 2)).reshape(self.ysize, 1)
        self.xlocs += xoffsets
        # get the mask for the cylinder
        self.cylinder_mask = (self.xlocs-self.xsize/4)**2 + (self.ylocs-self.ysize/2)**2 < (self.ysize/4)**2

    def drift_lattice(self):
        ''' Applies drift to the each node in the lattice. 
        
        Returns
        -------
        np.array
            A new lattice that has had it's velocities drifted.
        '''
        # creates arrays to define the drifts in terms of indices for each node
        newF = np.zeros_like(self.F)
        d_x_eveny = np.array([0,1,0,-1,-1,-1,0])
        d_x_oddy =  np.array([0,1,1,0,-1,0,1])
        d_y = np.array([0,0,-1,-1,0,1,1])
        xidxs = np.arange(self.xsize)
        # drift the even rows
        for y in range(0, self.ysize, 2):
            for i, dy, dx in zip(range(7), d_y, d_x_eveny):
                newF[(y-dy) % self.ysize, np.roll(xidxs,-dx,0), i] = self.F[y, xidxs, i]
        
        # drift the odd rows
        for y in range(1,self.ysize,2):
            for i, dy, dx in zip(range(7), d_y, d_x_oddy):
                newF[(y-dy) % self.ysize, np.roll(xidxs,-dx,0), i] = self.F[y, xidxs, i]

        return newF

    def update_frame(self):
        ''' Updates the frame. '''
        newArr = self.drift_lattice()
        for y in range(self.ysize):
            for x in range(self.xsize):
                if self.arr[y,x,0]:
                    newArr[y,x,:] = self.arr[y,x,[0,1,5,6,7,2,3,4]]
                elif self.arr[y,x] in UPDATE_MAP:
                    newArr[y,x] = UPDATE_MAP[self.arr[y,x]]
                else:
                    newArr[y,x] = self.arr[y,x]

    def get_pressure_frame(self):
        ''' Get a frame of the pressure values across the lattice.
        '''
        pass


        