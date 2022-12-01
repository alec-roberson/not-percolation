import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os
import json
import shutil
import cv2

def get_circle_plug_map(ydim, xdim, loc, num_per_col, num_col, rad, col_spacing, packing_method):
    # copute vertical spacing, calculate radius in pixels
    vert_spacing = int(ydim/num_per_col)
    rad = int(rad*vert_spacing/2)
    # convert col spacing and xloc to pixels
    if packing_method == 'standard':
        col_spacing = int(col_spacing*vert_spacing)
    elif packing_method == 'tight':
        col_spacing = int(2*rad*col_spacing)
    else:
        raise ValueError(f'Invalid packing method "{packing_method}"')
    xloc = int(xdim*loc)
    # calculate first row x position
    xloc0 = xloc - int((num_col-1)*col_spacing/2)
    # initialize output array
    arr = np.zeros((ydim, xdim))
    for i in range(num_col):
        # calculate x position
        x = xloc0 + i*col_spacing
        # determine first position and such based on even/odd cols
        if i % 2 == 0:
            yloc0 = int(vert_spacing/2)
            num_circs = num_per_col
        else:
            yloc0 = 0
            num_circs = num_per_col + 1
        # draw circles
        for j in range(num_circs):
            y = yloc0 + j*vert_spacing
            arr = cv2.circle(arr, (x, y), rad, 1, -1)
    return np.array(arr, dtype=bool).reshape(ydim, xdim)

def get_square_plug_map(ydim, xdim, loc, num_per_col, num_col, a, col_spacing):
    # copute vertical spacing, calculate a in pixels
    vert_spacing = int(ydim/num_per_col)
    a = int(a*vert_spacing/2)
    # convert col spacing and xloc to pixels
    col_spacing = int(col_spacing*vert_spacing)
    xloc = int(xdim*loc)
    # calculate first row x position
    xloc0 = xloc - int((num_col-1)*col_spacing/2)
    # initialize output array
    arr = np.zeros((ydim, xdim))
    for i in range(num_col):
        # calculate x position
        x = xloc0 + i*col_spacing
        # determine first position and such based on even/odd cols
        if i % 2 == 0:
            yloc0 = int(vert_spacing/2)
            num_circs = num_per_col
        else:
            yloc0 = 0
            num_circs = num_per_col + 1
        # draw circles
        for j in range(num_circs):
            y = yloc0 + j*vert_spacing
            arr = cv2.rectangle(arr, (x-a, y-a), (x+a,y+a), 1, -1)
    return np.array(arr, dtype=bool).reshape(ydim, xdim)

def get_triangle_plug_map(ydim, xdim, loc, num_per_col, num_col, a, col_spacing):
    # copute vertical spacing, calculate a in pixels
    vert_spacing = int(ydim/num_per_col)
    a = int(a*vert_spacing/2)
    # convert col spacing and xloc to pixels
    col_spacing = int(col_spacing*vert_spacing)
    xloc = int(xdim*loc)
    # calculate first row x position
    xloc0 = xloc - int((num_col-1)*col_spacing/2)
    # initialize output array
    arr = np.zeros((ydim, xdim))
    for i in range(num_col):
        # calculate x position
        x = xloc0 + i*col_spacing
        # determine first position and such based on even/odd cols
        if i % 2 == 0:
            yloc0 = int(vert_spacing/2)
            num_circs = num_per_col
        else:
            yloc0 = 0
            num_circs = num_per_col + 1
        # draw circles
        for j in range(num_circs):
            y = yloc0 + j*vert_spacing
            print([(x-a, y-a), (x+a, y), (x-a, y+a)])
            arr = cv2.fillConvexPoly(arr, np.array([(x-a, y-a), (x+a, y), (x-a, y+a)]), (1,))
    return np.array(arr, dtype=bool).reshape(ydim, xdim)

def get_smiley_plug_map(ydim, xdim, loc, **kwargs):
    # convert loc to pixels
    xloc = int(xdim*loc)
    # initialize output array
    arr = np.zeros((ydim, xdim))
    # draw smile
    arr = cv2.ellipse(arr, (xloc, int(ydim/2)), (int(ydim/4), int(ydim/4)), 0, 30, 150, 1, int(ydim/20))
    # draw eyes
    arr = cv2.circle(arr, (xloc-int(ydim/6), int(ydim/3)), int(ydim/10), 1, -1)
    arr = cv2.circle(arr, (xloc+int(ydim/6), int(ydim/3)), int(ydim/10), 1, -1)
    return np.array(arr, dtype=bool).reshape(ydim, xdim)

def get_plug_map(ydim, xdim, plug_type, **kwargs):
    if plug_type == 'circle':
        return get_circle_plug_map(ydim, xdim, **kwargs)
    elif plug_type == 'square':
        return get_square_plug_map(ydim, xdim, **kwargs)
    elif plug_type == 'triangle':
        return get_triangle_plug_map(ydim, xdim, **kwargs)
    elif plug_type == 'smiley':
        return get_smiley_plug_map(ydim, xdim, **kwargs)
    elif plug_type == 'none':
        return None

class FlowTube:
    def __init__(self, ydim, xdim, plug_map=None, tau=0.6, density=100, randomness=0.2, avg_vel=2.5, start_param=1, inflow_vel=2, inflow_density=100, inflow_noise=0.1, outflow_density=50, plot_every=1):
        # set global variables
        self.ydim = ydim
        self.xdim = xdim
        self.tau = tau
        self.avg_vel = avg_vel
        self.inflow_vel = inflow_vel
        self.inflow_density = inflow_density
        self.inflow_noise = inflow_noise
        self.outflow_density = outflow_density
        self.plot_every = plot_every
        self.pbar = None

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

        # add plug
        if not plug_map is None:
            self.boundaries += plug_map

        # add padding to boundaries
        self.padded_boundaries = np.concatenate([
            # self.boundaries[:,0:1],
            self.boundaries[:,0:1], 
            self.boundaries, 
            # self.boundaries[:,-1:], 
            self.boundaries[:,-1:]], axis=1)

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
        num_nodes = np.sum(1-self.boundaries, axis=0)
        avg_rho = np.sum(self.F * ~self.boundaries.reshape(self.ydim, self.xdim, 1), axis=(0,2))/num_nodes
        avg_x_flow = np.sum((self.F * ~self.boundaries.reshape(self.ydim, self.xdim, 1))*self.vel_x, axis=(0,2))/num_nodes
        return avg_x_flow / avg_rho

    def init_anim(self):
        self.fig, ((self.v_ax, self.xv_ax), (self.d_ax, self.xd_ax)) = plt.subplots(2,2)
        plt.tight_layout(pad=1.4)
        
        self.v_ax.set_title('Velocity Heat Map')
        self.v_map = self.v_ax.imshow(self.get_velocity_frame())

        self.d_ax.set_title('Density Heat Map')
        self.d_map = self.d_ax.imshow(self.get_density_frame())

        self.xv_ax.set_title('X Velocity')
        self.xv_graph, = self.xv_ax.plot(self.get_x_velocity())
        
        self.xd_ax.set_title('X Density')
        self.xd_graph, = self.xd_ax.plot(self.get_x_density())

        return [self.v_map, self.d_map, self.xv_graph, self.xd_graph]
        
    def anim(self, i):
        for _ in range(self.plot_every):
            self.update_frame()
        
        if self.pbar is not None:
            self.pbar.update(self.plot_every)

        self.v_map.set_data(self.get_velocity_frame())
        self.v_map.autoscale()

        self.d_map.set_data(self.get_density_frame())
        self.d_map.autoscale()

        xv = self.get_x_velocity()
        xv_mid = (np.min(xv) + np.max(xv))/2
        xv_range = np.max(xv) - np.min(xv)
        xv_range = max(xv_range, 1e-10)
        self.xv_ax.set_ylim(xv_mid - xv_range * 0.6, xv_mid + xv_range * 0.6)
        self.xv_graph.set_ydata(xv)

        xd = self.get_x_density()
        xd_mid = (np.min(xd) + np.max(xd))/2
        xd_range = np.max(xd) - np.min(xd)
        xd_range = max(xd_range, 1e-10)
        self.xd_ax.set_ylim(xd_mid - xd_range * 0.6, xd_mid + xd_range * 0.6)
        self.xd_graph.set_ydata(xd)

        self.v_ax.figure.canvas.draw()
        self.d_ax.figure.canvas.draw()
        self.xv_ax.figure.canvas.draw()
        self.xd_ax.figure.canvas.draw()

        return [self.v_map, self.d_map, self.xd_graph, self.xv_graph]

    def init_pbar(self, num_steps):
        self.pbar = tqdm(total=num_steps)

    def close_pbar(self):
        self.pbar.close()
    
    def anim_loop(self, N, plot_every=1, fps=30):
        velfig = plt.subplot(2, 2, 1)
        velfig.set_title('Velocity Heat Map')
        velim = velfig.imshow(self.get_velocity_frame())
        rhofig = plt.subplot(2, 2, 3)
        rhofig.set_title('Density Heat Map')
        rhoim = rhofig.imshow(self.get_density_frame())
        # curlfig = plt.subplot(3, 2, 5)
        # curlfig.set_title('Curl Heat Map')
        # curlim = curlfig.imshow(self.get_curl_frame(), cmap='bwr')
        xd_fig = plt.subplot(2,2,2)
        xd_fig.set_title('X Density')
        xd_l, = xd_fig.plot(self.get_x_density())
        xv_fig = plt.subplot(2,2,4)
        xv_fig.set_title('X Velocity')
        xv_l, = xv_fig.plot(self.get_x_velocity())
        plt.tight_layout(pad=1)
        for i in range(N):
            self.update_frame()
            if i % plot_every == 0:
                velim.set_data(self.get_velocity_frame())
                velim.autoscale()

                rhoim.set_data(self.get_density_frame())
                rhoim.autoscale()

                # cframe = self.get_curl_frame()
                # curlim.set_data(cframe)
                # l = np.max(np.abs(cframe))
                # curlim.set_clim(-l, l)

                xd = self.get_x_density()
                xd_l.set_ydata(xd)
                xd_mid = (np.min(xd) + np.max(xd))/2
                xd_range = np.max(xd) - np.min(xd)
                xd_fig.set_ylim(xd_mid - xd_range * 0.6, xd_mid + xd_range * 0.6)
                
                xv = self.get_x_velocity()
                xv_l.set_ydata(xv)
                xv_mid = (np.min(xv) + np.max(xv))/2
                xv_range = np.max(xv) - np.min(xv)
                xv_fig.set_ylim(xv_mid - xv_range * 0.6, xv_mid + xv_range * 0.6)

                plt.pause(1/fps)

if __name__ == '__main__':
    # Check input
    if not os.path.isfile('./variables.json'):
        print('Error: variables.json not found')
        exit(1)
    # Read variables
    with open('./variables.json', 'r') as f:
        invars = json.load(f)
        cfg = invars['config']
        args = invars['args']
        plug_args = invars['plug']
    # Check output
    if os.path.isdir(cfg['output_dir']):
        print('Error: output directory already exists')
        exit(1)
    # Set up the simulation
    plug_map = get_plug_map(args['ydim'], args['xdim'], **plug_args)
    ft = FlowTube(**args, plug_map=plug_map)
    gifwriter = animation.PillowWriter(fps=cfg['fps'])
    ft.init_anim()
    num_steps = cfg['num_steps']
    num_frames = num_steps//ft.plot_every
    ft.init_pbar(num_steps)
    anim = animation.FuncAnimation(ft.fig, ft.anim, frames=num_frames, blit=True)
    
    # save stuff
    os.makedirs(cfg['output_dir'])
    shutil.copy('./variables.json', os.path.join(cfg['output_dir'], 'variables.json'))
    anim.save(os.path.join(cfg['output_dir'], 'animation.gif'), writer=gifwriter)
