import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class FlowTube:
    def __init__(self, ydim, xdim, tau=0.6, density=100, randomness=0.2, avg_vel=2.5, start_param=1, inflow_vel=2, inflow_density=100, inflow_noise=0.1, outflow_density=50):
        # set global variables
        self.ydim = ydim
        self.xdim = xdim
        self.tau = tau
        self.avg_vel = avg_vel
        self.inflow_vel = inflow_vel
        self.inflow_density = inflow_density
        self.inflow_noise = inflow_noise
        self.outflow_density = outflow_density

        # get physical positions of particles
        self.xlocs, self.ylocs = np.meshgrid(range(self.xdim), range(self.ydim))

        # velocity conversion variables and weights
        self.nvels = 9
        self.vel_y = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
        self.vel_x = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
        self.weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

        # create lattice & setup initial conditions
        self.F = np.ones((self.ydim, self.xdim, self.nvels)) * start_param
        self.F += randomness * np.random.randn(ydim, xdim, self.nvels)
        self.F[:, :, 3] += self.avg_vel
        rho = np.sum(self.F, axis=2).reshape(self.ydim, self.xdim, 1)
        self.F = self.F * density/rho

        # create boundary mapping
        self.boundaries = np.zeros((ydim, xdim), dtype=bool)
        # self.boundaries[:, 0] = True
        # self.boundaries[:, -1] = True
        self.boundaries[0, :] = True
        self.boundaries[-1, :] = True

        # add cylinder
        for x, y in zip([self.xdim/4]*4+[self.xdim/4+5*self.ydim / 32]*3+[self.xdim/4+5*self.ydim / 16]*4, [1/8*self.ydim, 3/8*self.ydim, 5/8*self.ydim, 7/8*self.ydim]+[1/4*self.ydim, 1/2*self.ydim, 3/4*self.ydim]+[1/8*self.ydim, 3/8*self.ydim, 5/8*self.ydim, 7/8*self.ydim]):
            self.boundaries += (self.xlocs - x) ** 2 + (self.ylocs - y) ** 2 < (3*self.ydim / 32) ** 2

        # add padding to boundaries
        self.padded_boundaries = np.concatenate([
            # self.boundaries[:,0:1],
            self.boundaries[:,0:1], 
            self.boundaries, 
            # self.boundaries[:,-1:], 
            self.boundaries[:,-1:]], axis=1)

        self.F[self.boundaries] = 1e-10

    def get_source(self):        
        # set the input flow

        # correct the input flow density

        source = np.zeros((self.ydim, self.nvels))
        source[:,:] = self.inflow_noise * np.abs(np.random.randn(self.ydim, self.nvels))
        source[:,[3]] += self.inflow_vel * np.ones((self.ydim, 1))
        rho_inflow = np.sum(source, axis=1).reshape(self.ydim, 1)
        source *= self.inflow_density / rho_inflow

        # apply a sinusoidal weighting
        weights = 2*np.sin(np.linspace(0,1,self.ydim)*np.pi).reshape(self.ydim, 1)**2

        source *= weights
        source = source.reshape(self.ydim, 1, self.nvels)
        source = np.concatenate([source, source], axis=1)
        return source
    
    def get_sink(self):
        # the sink will just average the values from the N columns
        N = 20
        M = 10
        sink = np.mean(self.F[:,-1-N:-1-M,:], axis=1).reshape(self.ydim, 1, self.nvels)
        if not self.outflow_density is None:
            rho_sink = np.sum(sink, axis=2).reshape(self.ydim, 1, 1)
            sink *= self.outflow_density / rho_sink
        sink = np.concatenate([sink, sink], axis=1)
        return sink

    def add_source_and_sink(self):
        # add padding, source on left sink on right
        self.F = np.concatenate([self.get_source(), self.F, self.get_sink()], axis=1)

    def strip_one_padding(self):
        # remove padding, source on left sink on right
        self.F = self.F[:, 1:-1, :]

    def apply_drift(self):

        # drift velocities
        for i, cy, cx in zip(range(self.nvels), self.vel_y, self.vel_x):
            self.F[:, :, i] = np.roll(self.F[:, :, i], cx, axis = 1)
            self.F[:, :, i] = np.roll(self.F[:, :, i], cy, axis = 0)

        # get rid of any wrapping-around
        self.strip_one_padding()

    def update_frame(self):
        ''' Updates the frame.
        '''
        # add padding
        self.add_source_and_sink()
        
        # do drift
        self.apply_drift()

        # reflect velocities along boundaries
        boundaryF = self.F[self.padded_boundaries, :]
        boundaryF = boundaryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        

        # fluid variables
        rho = np.sum(self.F, axis=2)
        uy = np.sum(self.F * self.vel_y, axis=2) / rho
        ux = np.sum(self.F * self.vel_x, axis=2) / rho

        # calculate equilibrium and apply collisions
        Feq = np.zeros_like(self.F)
        for i, cy, cx, w in zip(range(self.nvels), self.vel_y, self.vel_x, self.weights):
            Feq[:, :, i] = w * rho * (1 + 3 * (cx * ux + cy * uy) + 9/2 * (cx * ux + cy * uy) ** 2 - 3/2 * (ux ** 2 + uy ** 2))
        self.F += -(1/self.tau)*(self.F - Feq)
        # self.F[:,1:-1] += -(1/self.tau)*(self.F[:,1:-1] - Feq[:,1:-1])

        # apply boundary conditions
        self.F[self.padded_boundaries, :] = boundaryF

        # tidy up F (no negative values)
        self.F[self.F < 0] = 1e-10

        # remove padding
        self.strip_one_padding()

    def get_velocity_frame(self):
        ''' Gets the frame of velocities. '''
        rho = np.sum(self.F, axis=2)
        ux = np.sum(self.F * self.vel_x, axis=2) / rho
        uy = np.sum(self.F * self.vel_y, axis=2) / rho
        vels = np.sqrt(ux**2 + uy**2)
        vels[self.boundaries] = 0
        return vels
    
    def get_density_frame(self):
        rho = np.sum(self.F, axis=2)
        rho[self.boundaries] = 0
        return rho
    
    def get_curl_frame(self):
        rho = np.sum(self.F, axis=2)
        ux = np.sum(self.F * self.vel_x, axis=2) / rho
        uy = np.sum(self.F * self.vel_y, axis=2) / rho
        ux[self.boundaries] = 0
        uy[self.boundaries] = 0
        curl = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        curl[self.boundaries] = 0
        return curl[:,1:-1]
    
    def total(self):
        F = self.F * ~self.boundaries.reshape(self.ydim, self.xdim, 1)
        return np.sum(F), np.max(F), np.min(F)
    
    def get_x_density(self):
        return np.sum(self.F * ~self.boundaries.reshape(self.ydim, self.xdim, 1), axis=(0,2))/np.sum(1-self.boundaries, axis=0)

    def get_x_velocity(self):
        return np.sum(
            (self.F * ~self.boundaries.reshape(self.ydim, self.xdim, 1))*self.vel_x, axis=(0,2))/np.sum(1-self.boundaries, axis=0)

    def anim_loop(self, N, plot_every=1, fps=30):
        velfig = plt.subplot(5, 1, 1)
        velfig.set_title('Velocity Heat Map')
        velim = velfig.imshow(self.get_velocity_frame())
        rhofig = plt.subplot(5, 1, 2)
        rhofig.set_title('Density Heat Map')
        rhoim = rhofig.imshow(self.get_density_frame())
        curlfig = plt.subplot(5, 1, 3)
        curlfig.set_title('Curl Heat Map')
        curlim = curlfig.imshow(self.get_curl_frame(), cmap='bwr')
        xd_fig = plt.subplot(5,1,4)
        xd_fig.set_title('X Density')
        xd_fig.plot(self.get_x_density())
        xv_fig = plt.subplot(5,1,5)
        xv_fig.set_title('X Density')
        xv_fig.plot(self.get_x_velocity())
        for i in range(N):
            self.update_frame()
            if i % plot_every == 0:
                velim.set_data(self.get_velocity_frame())
                velim.autoscale()

                rhoim.set_data(self.get_density_frame())
                rhoim.autoscale()

                cframe = self.get_curl_frame()
                curlim.set_data(cframe)
                l = np.max(np.abs(cframe))
                curlim.set_clim(-l, l)

                xd_fig.clear()
                xd_fig.plot(self.get_x_density())
                xd_fig.autoscale()
                
                xv_fig.clear()
                xv_fig.plot(self.get_x_velocity())
                xv_fig.autoscale()
                plt.pause(1/fps)
                print(self.total())

if __name__ == '__main__':
    ft = FlowTube(100, 600,
        tau=5, 
        density=10, 
        randomness=0.1,
        avg_vel=1,
        start_param=1, 
        inflow_vel=1,
        inflow_noise=0.1, 
        inflow_density=10, 
        outflow_density=None)
    ft.anim_loop(10000, 10, 15)