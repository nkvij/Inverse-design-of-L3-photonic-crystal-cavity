import numpy as np
import legume
legume.set_backend('autograd')
import matplotlib.pyplot as plt
import os

import autograd.numpy as npa
from autograd import value_and_grad
from scipy.optimize import minimize
from numba import njit
import time as time
import autograd
from joblib import Parallel, delayed

autograd.extend.defvjp(
    autograd.numpy.asarray,
    lambda ans, *args, **kw: lambda g: g
)
def L3_params_930():
    '''
    Wrong thickness of the membrane considered. Here it is 180 nm. See L3_params_936() for correct numbers.
    '''
    
    hole_radius = 0.3
    slab_thickness = 0.782
    refractive_index = 3.55
    x_shift = 0.182
    
    return hole_radius, slab_thickness, refractive_index, x_shift
    
def L3_params_936():
    
    hole_radius = 0.313
    slab_thickness = 0.826
    refractive_index = 3.55
    x_shift = 0.178
    
    return hole_radius, slab_thickness, refractive_index, x_shift

def L3_params_440():
    
    hole_radius = 0.3
    slab_thickness = 85/145
    refractive_index = 2.75
    x_shift = 0.1
    
    return hole_radius, slab_thickness, refractive_index, x_shift

def L3_params_520():
    
    hole_radius = 0.3
    slab_thickness = 1.081
    refractive_index = 2
    x_shift = 0.26
    
    return hole_radius, slab_thickness, refractive_index, x_shift

def L3_params_1550():
    
    hole_radius = 0.25
    slab_thickness = 0.55
    refractive_index = 3.4757
    
    return hole_radius, slab_thickness, refractive_index

def phc_cavity_holes_array(cavity_name, Nx, Ny):
    
    x_pos, y_pos = [], []
    nx, ny = Nx//2 + 1, Ny//2 + 1
    
    if(cavity_name == 'L3'):
        for iy in range(ny):
            for ix in range(nx):
                if(ix == 0 and iy == 0 or ix == 1 and iy == 0): # skip holes for L3 cavity
                    continue
                else:
                    x_pos.append(ix + (iy%2)*0.5)
                    y_pos.append(iy*np.sqrt(3)/2)
   
    if(cavity_name == 'H1'):
        for iy in range(ny):
            for ix in range(nx):
                if(ix == 0 and iy == 0): # skip holes for L3 cavity
                    continue
                else:
                    x_pos.append(ix + (iy%2)*0.5)
                    y_pos.append(iy*np.sqrt(3)/2)
        
    return x_pos, y_pos

def design_phc(Nx, Ny, x_pos, y_pos, hole_radius, slab_thickness, refractive_index, dx, dy):
    
    # Initialize a lattice and PhC
    lattice = legume.Lattice([Nx, 0], [0, Ny*np.sqrt(3)/2])
    L3_phc = legume.PhotCryst(lattice)

    # Add a layer to the PhC
    L3_phc.add_layer(d=slab_thickness, eps_b=refractive_index**2)
    
    # Get the position of holes for L3 cavity
    nx, ny = Nx//2 + 1, Ny//2 + 1

    # Apply holes symmetrically in the four quadrants
    for ic, x in enumerate(x_pos):
        yc = y_pos[ic] if y_pos[ic] == 0 else y_pos[ic] + dy[ic]
        xc = x if x == 0 else x_pos[ic] + dx[ic]
        L3_phc.add_shape(legume.Circle(x_cent=xc, y_cent=yc, r=hole_radius))
        if nx-0.6 > x_pos[ic] > 0 and (ny-1.1)*np.sqrt(3)/2 > y_pos[ic] > 0:
            L3_phc.add_shape(legume.Circle(x_cent=-xc, y_cent=-yc, r=hole_radius))
        if nx-1.6 > x_pos[ic] > 0:
            L3_phc.add_shape(legume.Circle(x_cent=-xc, y_cent=yc, r=hole_radius))
        if (ny-1.1)*np.sqrt(3)/2 > y_pos[ic] > 0 and nx-1.1 > x_pos[ic]:
            L3_phc.add_shape(legume.Circle(x_cent=xc, y_cent=-yc, r=hole_radius))
            
    return L3_phc

def design_H1_cavity(Nx, Ny, x_pos, y_pos, hole_radius, slab_thickness, refractive_index, dx, dy):
    
    # Initialize a lattice and PhC
    lattice = legume.Lattice([Nx, 0], [0, Ny*np.sqrt(3)/2])
    L3_phc = legume.PhotCryst(lattice)

    # Add a layer to the PhC
    L3_phc.add_layer(d=slab_thickness, eps_b=refractive_index**2)
    
    # Get the position of holes for L3 cavity
    nx, ny = Nx//2 + 1, Ny//2 + 1

    # Apply holes symmetrically in the four quadrants
    for ic, x in enumerate(x_pos):
        yc = y_pos[ic] if y_pos[ic] == 0 else y_pos[ic] + dy[ic]
        xc = x if x == 0 else x_pos[ic] + dx[ic]
        L3_phc.add_shape(legume.Circle(x_cent=xc, y_cent=yc, r=hole_radius))
        if nx-0.6 > x_pos[ic] > 0 and (ny-1.1)*np.sqrt(3)/2 > y_pos[ic] > 0:
            L3_phc.add_shape(legume.Circle(x_cent=-xc, y_cent=-yc, r=hole_radius))
        if nx-1.6 > x_pos[ic] > 0:
            L3_phc.add_shape(legume.Circle(x_cent=-xc, y_cent=yc, r=hole_radius))
        if (ny-1.1)*np.sqrt(3)/2 > y_pos[ic] > 0 and nx-1.1 > x_pos[ic]:
            L3_phc.add_shape(legume.Circle(x_cent=xc, y_cent=-yc, r=hole_radius))
            
    return L3_phc

def gme_cavity(phc, Nx, Ny, gaussian_width, gmax, kpoints, options):

    # Initialize GME
    gme = legume.GuidedModeExp(phc, gmax=gmax)

    # Solve for the real part of the frequencies
    gme.run(kpoints=kpoints, **options)

    # Calculate imaginary frequencies that correspond to cavity losses
    f_imag = []
    # Calculate the directionality of emission
    direction_ratio = []
    
    indmode = Nx*Ny
    
    kx_array = []
    ky_array = []
    rad_coupling_array = []
    
    if(gaussian_width != 0):
        # Gaussian far-field
        for i in range(kpoints[0, :].size):
            (freq_im, rad_coup, rad_gvec) = gme.compute_rad(i, [indmode])
            f_imag.append(freq_im)
            
            kx_array.extend(rad_gvec['l'][0][0] + kpoints[0][i])
            ky_array.extend(rad_gvec['l'][0][1] + kpoints[1][i])
            rad_coupling_array.extend(npa.abs(rad_coup['l_te'][0])**2 + npa.abs(rad_coup['l_tm'][0])**2)
        
        kx_array = npa.array(kx_array)/gmax
        ky_array = npa.array(ky_array)/gmax
        rad_coupling_array = npa.array(rad_coupling_array)
            
        overlap = calculate_gaussian_overlap(kx_array, ky_array, rad_coupling_array, gaussian_width)
        direction_ratio.append(overlap)
                
    elif(gaussian_width == 0):
        # Directional far-field
        for i in range(kpoints[0, :].size):
            (freq_im, rad_coup, rad_gvec) = gme.compute_rad(i, [indmode])
            f_imag.append(freq_im)
            
            kx_array.extend(rad_gvec['l'][0][0] + kpoints[0][i])
            ky_array.extend(rad_gvec['l'][0][1] + kpoints[1][i])
            rad_coupling_array.extend(npa.abs(rad_coup['l_te'][0])**2 + npa.abs(rad_coup['l_tm'][0])**2)
        
        kx_array = npa.array(kx_array)/gmax
        ky_array = npa.array(ky_array)/gmax
        rad_coupling_array = npa.array(rad_coupling_array)
            
        direction_ratio.append(calculate_directionality(kx_array, ky_array, rad_coupling_array))
                    
    return (gme, npa.array(f_imag), npa.array(direction_ratio), indmode)
    
def gme_cavity_dot(phc, gaussian_width, eigvec, gmax, kpoints, gme_options):
    
    gmax = gmax + 0.01 # to avoid warnings
    # Iniialize GME
    gme = legume.GuidedModeExp(phc, gmax = gmax)
    
    # rum GME solver
    gme.run(kpoints = kpoints, **gme_options)
    
    # Figure out the index of the fundamental mode
    dot_product = npa.abs(npa.dot(eigvec, npa.asarray(gme.eigvecs[0])))
    
    kx_array = []
    ky_array = []
    rad_coupling_array = []
    
    try:
        indmode = npa.nonzero(dot_product > 0.3)[0][0]
        
        # Calculate imaginary frequencies that correspond to cavity losses
        f_imag = []
        # Calculate the directionality of emission
        direction_ratio = []
        
        if(gaussian_width != 0):
            # Gaussian far-field
            for i in range(kpoints[0, :].size):
                (freq_im, rad_coup, rad_gvec) = gme.compute_rad(i, [indmode])
                f_imag.append(freq_im)
                
                kx_array.extend(rad_gvec['l'][0][0] + kpoints[0][i])
                ky_array.extend(rad_gvec['l'][0][1] + kpoints[1][i])
                rad_coupling_array.extend(npa.abs(rad_coup['l_te'][0])**2 + npa.abs(rad_coup['l_tm'][0])**2)
            
            kx_array = npa.array(kx_array)
            ky_array = npa.array(ky_array)
            rad_coupling_array = npa.array(rad_coupling_array)
                
            overlap = calculate_gaussian_overlap(kx_array, ky_array, rad_coupling_array, gaussian_width)
            direction_ratio.append(overlap)
                
        elif(gaussian_width == 0):
            # Directional far-field
            for i in range(kpoints[0, :].size):
                (freq_im, rad_coup, rad_gvec) = gme.compute_rad(i, [indmode])
                f_imag.append(freq_im)
                
                kx_array.extend(rad_gvec['l'][0][0] + kpoints[0][i])
                ky_array.extend(rad_gvec['l'][0][1] + kpoints[1][i])
                rad_coupling_array.extend(npa.abs(rad_coup['l_te'][0])**2 + npa.abs(rad_coup['l_tm'][0])**2)
            
            kx_array = npa.array(kx_array)
            ky_array = npa.array(ky_array)
            rad_coupling_array = npa.array(rad_coupling_array)
                
            direction_ratio.append(calculate_directionality(kx_array, ky_array, rad_coupling_array))
                    
        else:
            raise ValueError("Gaussian width should be > 0")
        
        return (gme, npa.array(f_imag), npa.array(direction_ratio), indmode)
    except:
        raise ValueError(f"Mode not found! max(dot_product) = {np.max(dot_product)}")
    
def gme_cavity_field(phc, num_modes, slab_thickness, gmax, kpoints, options):
    
    # Initialize GME
    gme = legume.GuidedModeExp(phc, gmax=gmax)

    # Solve for the real part of the frequencies
    gme.run(kpoints=kpoints, **options)

    # Calculate imaginary frequencies that correspond to cavity losses
    f_imag_array = []
    
    # Calculate the directionality of emission
    direction_ratio_array = []
    
    field_maxima_array = []
    
    for j in range(num_modes):
        Ey = gme.get_field_xy('e', kind=0, mind=j, z=slab_thickness/2, component='y')[0]['y']
        (xshape, yshape) = Ey.shape
        
        field_maxima_array.append(npa.square(npa.abs(Ey[xshape//2, yshape//2])))
        
    field_maxima_array = npa.array(field_maxima_array)
    
    indmode = npa.argmax(field_maxima_array)
        
    for i in range(kpoints[0, :].size):
        (freq_im, rad_coup, rad_gvec) = gme.compute_rad(i, [indmode])
        direction_ratio_array.append(calculate_directionality(rad_coup, rad_gvec))
        f_imag_array.append(freq_im)
        
    return (gme, npa.array(f_imag_array), npa.array(direction_ratio_array), indmode)
    
def get_kpoints(Nx, Ny, nkx, nky):
    # sample nkx and nky points in {kx, ky} space in a uniform grid
    # spacing between two reciprocal vectors is 2*pi/N

    kx = npa.linspace(0, (nkx-1)/nkx* 2*npa.pi/Nx, nkx)
    ky = npa.linspace(0, (nky-1)/nky* 2*npa.pi/Ny/npa.sqrt(3)*2, nky)
    kxg, kyg = npa.meshgrid(kx, ky)
    kxg = kxg.ravel()
    kyg = kyg.ravel()
    
    kpoints = npa.vstack((kxg, kyg))
    return kpoints

def calculate_directionality(kx_array, ky_array, rad_coup):

    index_00 = npa.argmin(npa.abs(kx_array) + npa.abs(ky_array))
    rad_00 = rad_coup[index_00]
    rad_tot = npa.sum(rad_coup)
    
    return rad_00/rad_tot

def calculate_circ_directionality(rad_coup, rad_gvec):
    x = npa.asarray(rad_gvec['l'][0][0])
    y = npa.asarray(rad_gvec['l'][0][1])
    
    k = npa.sqrt(x**2 + y**2)
    
    inds = npa.argwhere(k < 0.7)
    
    rad_tot = npa.sum(npa.abs(rad_coup['l_te'][0])**2) + npa.sum(npa.abs(rad_coup['l_tm'][0])**2)
    rad_00 = npa.abs(rad_coup['l_te'][0][inds])**2/rad_tot + npa.abs(rad_coup['l_tm'][0][inds])**2/rad_tot

    return npa.sum((rad_00-1/9)**2)
    
def calculate_gaussian_directionality(rad_coup, rad_gvec, gaussian_waist):
    
    x = npa.asarray(rad_gvec['l'][0][0])
    y = npa.asarray(rad_gvec['l'][0][1])
    
    k = npa.max(npa.sqrt(x**2 + y**2))
    x,y = x/k, y/k
    
    target_gaussian = npa.exp(-(x**2 + y**2)*gaussian_waist)
    target_gaussian_norm = target_gaussian/npa.sqrt(npa.sum(target_gaussian**2))
    
    rad_tot = npa.sum(npa.abs(rad_coup['l_te'][0])**2) + npa.sum(npa.abs(rad_coup['l_tm'][0])**2)
    rad = (npa.abs(rad_coup['l_te'][0])**2 + npa.abs(rad_coup['l_tm'][0])**2)/npa.sqrt(rad_tot)
    
    gaussian_directionality = npa.sum(((rad - target_gaussian_norm)/target_gaussian_norm)**2)
    
    return gaussian_directionality
    
def calculate_gaussian_overlap(kx_array, ky_array, rad_coup_array, gaussian_waist):
    
    k_norm = npa.sqrt(npa.max(kx_array**2 + ky_array**2))
    kx_array = kx_array/k_norm
    ky_array = ky_array/k_norm
    
    # kx and ky are in sine space: kx = sin(theta)*sin(phi), ky = sin(theta)*cos(phi)
    sine_theta = npa.sqrt(kx_array**2 + ky_array**2)
    
    # define target gaussian array
    target_gaussian = 1/(2*npa.pi*gaussian_waist**2)*npa.exp(-(kx_array**2 + ky_array**2)/(2*gaussian_waist**2))
    target_gaussian_norm = target_gaussian/npa.sqrt(npa.sum(target_gaussian**2*sine_theta))
    
    # Normalise the radiative couplings array
    rad_norm = npa.sqrt(rad_coup_array)/npa.sqrt(npa.sum(rad_coup_array*sine_theta))
    
    # Calculate gaussian overlap in spherical coordinates
    overlap = npa.dot(rad_norm,target_gaussian_norm*sine_theta)**2
    
    return overlap

def gme_cavity_k(phc, gmax, options, kpoints, f_lb):

    # Iniialize GME
    gme = legume.GuidedModeExp(phc, gmax = gmax)
    
    # rum GME solver
    gme.run(kpoints = kpoints, **options)
    
    # Figure out the index of the fundamental mode (I know this from previous
    # simulations)
    indmode = npa.nonzero(gme.freqs[0, :] > f_lb)[0][0]
    
    # Calculate imaginary frequencies that correspond to cavity losses
    f_imag = []
    # Calculate the directionality of emission
    direction_ratio = []
    
    for i in range(kpoints[0, :].size):
        (freq_im, rad_coup, rad_gvec) = gme.compute_rad(i, [indmode])
        direction_ratio.append(calculate_directionality(rad_coup))
        f_imag.append(freq_im)
        
    return (gme, npa.array(f_imag), indmode, npa.array(direction_ratio))
    
def load_params_from_history(file_name, length):
    
    param_history = np.load(file_name, allow_pickle=True)
    param_history = param_history.reshape(param_history.shape[0]//(2*length), 2*length)
    parameters = param_history[-1]
    
    return parameters

def calculate_Q(freqs, f_imag, indmode):
    return freqs[0, indmode]/2/npa.mean(f_imag)

def get_chris_params():
    alattice = 0.4
    hole_radius = 0.25
    slab_thickness = 0.55
    refractive_index = 3.4757
    
    return alattice, hole_radius, slab_thickness, refractive_index

def design_L3_cavity_chris(Nx, Ny, x_pos, y_pos, hole_radius, slab_thickness, refractive_index):
    
    # Initialize a lattice and PhC
    lattice = legume.Lattice([Nx, 0], [0, Ny*np.sqrt(3)/2])
    L3_phc_chris = legume.PhotCryst(lattice)

    # Add a layer to the PhC
    L3_phc_chris.add_layer(d=slab_thickness, eps_b=refractive_index**2)
    
    for x, y in zip(x_pos, y_pos):
        L3_phc_chris.add_shape(legume.Circle(x_cent=x, y_cent=y, r=hole_radius))
        
    return L3_phc_chris

class Minimize_Neelesh(object):
    """Wrapping up custom and SciPy optimizers in a common class
    """
    def __init__(self, objective):

        self.objective = objective

        # Some internal variables
        self.iteration = 0
        self.of_list = []
        self.p_opt = []
        self.t_store = time.time()
        self.param_history = np.array([])
        self.t_elapsed_array = np.array([])

    @staticmethod
    def _get_value(x):
        """This is used when gradients are computed with autograd and the 
        objective function is an ArrayBox. Same function as in legume.utils, 
        but re-defined here so that this class could also be used independently 
        """
        if str(type(x)) == "<class 'autograd.numpy.numpy_boxes.ArrayBox'>":
            return x._value
        else:
            return x

    def _parse_bounds(self, bounds):
        """Parse the input bounds, which can be 'None', a list with two
        elements, or a list of tuples with 2 elements each
        """
        try:
            if bounds == None:
                return None
            elif not isinstance(bounds[0], tuple):
                if len(bounds) == 2:
                    return [tuple(bounds) for i in range(self.params.size)]
                else:
                    raise ValueError
            elif len(bounds) == self.params.size:
                if all([len(b) == 2 for b in bounds]):
                    return bounds
                else:
                    raise ValueError
            else:
                raise ValueError
        except:
            raise ValueError(
                "'bounds' should be a list of two elements "
                "[lb, ub], or a list of the same length as the number of "
                "parameters where each element is a tuple (lb, ub)")

    def _disp(self, t_elapsed):
        """Display information at every iteration
        """
        disp_str = "Epoch: %4d/%4d | Duration: %6.2f secs" % \
                        (self.iteration, self.Nepochs, t_elapsed)
        disp_str += " | Objective: %4e" % self.of_list[-1]
        if self.disp_p:
            disp_str += " | Parameters: %s" % self.params
        print(disp_str)

    def adam(self,
             pstart,
             Nepochs=50,
             bounds=None,
             disp_p=False,
             step_size=1e-2,
             beta1=0.9,
             beta2=0.999,
             args=(),
             pass_self=False,
             callback=None):
        """Performs 'Nepoch' steps of ADAM minimization with parameters 
        'step_size', 'beta1', 'beta2'

        Additional arguments:
        bounds          -- can be 'None', a list of two elements, or a 
            scipy.minimize-like list of tuples each containing two elements
            The 'bounds' are set abruptly after the update step by snapping the 
            parameters that lie outside to the bounds value
        disp_p          -- if True, the current parameters are displayed at 
            every iteration
        args            -- extra arguments passed to the objective function
        pass_self       -- if True, then the objective function should take
            of(params, args, opt), where opt is an instance of the Minimize 
            class defined here. Useful for scheduling
        Callback        -- function to call at every epoch; the argument that's
                            passed in is the current minimizer state
        """
        self.params = pstart
        self.bounds = self._parse_bounds(bounds)
        self.Nepochs = Nepochs
        self.disp_p = disp_p

        # Restart the counters
        self.iteration = 0
        self.t_store = time.time()
        self.of_list = []

        if pass_self == True:
            arglist = list(args)
            arglist.append(self)
            args = tuple(arglist)
            
        #def adam_call(iteration):
            
        for iteration in range(Nepochs):
            self.iteration += 1

            self.t_store = time.time()
            of, grad = value_and_grad(self.objective)(self.params, *args)
            t_elapsed = time.time() - self.t_store
            self.t_elapsed_array = np.append(self.t_elapsed_array, t_elapsed)

            self.of_list.append(self._get_value(of))
            self._disp(t_elapsed)

            if iteration == 0:
                mopt = np.zeros(grad.shape)
                vopt = np.zeros(grad.shape)

            (grad_adam, mopt, vopt) = self._step_adam(grad, mopt, vopt,
                                                      iteration, beta1, beta2)
            # Change parameters towards minimizing the objective
            if iteration < Nepochs - 1:
                self.params = self.params - step_size * np.exp(-2.3*iteration/Nepochs) * grad_adam

            self.param_history = np.append(self.param_history, np.array(self.params))
            
            temp_param_history = self.param_history.flatten()
            
            # if iteration%50 == 0:
            #     np.save("weights_520nm/temp_param_history.npy", np.asarray(temp_param_history))

            if bounds:
                #self.param = np.clip(self.param, -bounds, +bounds)
                lbs = np.array([b[0] for b in self.bounds])
                ubs = np.array([b[1] for b in self.bounds])
                self.params[self.params < lbs] = lbs[self.params < lbs]
                self.params[self.params > ubs] = ubs[self.params > ubs]

            if callback is not None:
                callback(self)

        return (self.params, self.of_list, self.param_history, self.t_elapsed_array)

    @staticmethod
    @njit(nopython = True, cache = True)
    def _step_adam(gradient,
                   mopt_old,
                   vopt_old,
                   iteration,
                   beta1,
                   beta2,
                   epsilon=1e-8):
        """Performs one step of Adam optimization
        """

        mopt = beta1 * mopt_old + (1 - beta1) * gradient
        mopt_t = mopt / (1 - beta1**(iteration + 1))
        vopt = beta2 * vopt_old + (1 - beta2) * (np.square(gradient))
        vopt_t = vopt / (1 - beta2**(iteration + 1))
        grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

        return (grad_adam, mopt, vopt)

    def lbfgs(self,
              pstart,
              Nepochs=50,
              bounds=None,
              disp_p=False,
              maxfun=15000,
              args=(),
              pass_self=False,
              res=False,
              callback=None):
        """Wraps the SciPy LBFGS minimizer in a way that displays intermediate
        information and stores intermediate values of the parameters and the
        objective function.

        Nepochs         -- Maximum number of iterations
        bounds          -- can be 'None', a list of two elements, or a 
            scipy.minimize-like list of tuples each containing two elements
            The 'bounds' are set abruptly after the update step by snapping the 
            parameters that lie outside to the bounds value
        disp_p          -- if True, the current parameters are displayed at 
            every iteration
        maxfun          -- Maximum number of function evaluations
        args            -- extra arguments passed to the objective function
        pass_self       -- if True, then the objective function should take
            of(params, args, opt), where opt is an instance of the Minimize 
            class defined here. Useful for scheduling
        res             -- if True, will also return the SciPy OptimizeResult
        callback        -- function to call at every epoch; the argument that's
                            passed in is the current minimizer state
        """

        self.params = pstart
        self.bounds = self._parse_bounds(bounds)
        self.Nepochs = Nepochs
        self.disp_p = disp_p

        # Restart the counters
        self.iteration = 0
        self.t_store = time.time()
        self.of_list = []

        # Get initial of value
        of = self.objective(self.params, *args)
        self.of_list.append(self._get_value(of))

        def of(params, *args, **kwargs):
            """Modify the objective function slightly to allow storing
            intermediate objective values without re-evaluating the function
            """
            if pass_self == True:
                arglist = list(args)
                arglist.append(self)
                args = tuple(arglist)

            out = value_and_grad(self.objective)(params, *args, **kwargs)
            self.of_last = self._get_value(out[0])
            return out

        def cb(xk):
            """Callback function for the SciPy minimizer
            """
            self.iteration += 1
            t_current = time.time()
            t_elapsed = t_current - self.t_store
            self.t_store = t_current

            self.of_list.append(self.of_last)
            self.params = xk
            self._disp(t_elapsed)

            # Call the custom callback function if any
            if callback is not None:
                callback(self)

        res_opt = minimize(of,
                           self.params,
                           args=args,
                           method='L-BFGS-B',
                           jac=True,
                           bounds=self.bounds,
                           tol=None,
                           callback=cb,
                           options={
                               'disp': False,
                               'maxcor': 10,
                               'ftol': 1e-8,
                               'gtol': 1e-5,
                               'eps': 1e-08,
                               'maxfun': maxfun,
                               'maxiter': Nepochs,
                               'iprint': -1,
                               'maxls': 20
                           })

        if res == False:
            return (res_opt.x, self.of_list)
        else:
            return (res_opt.x, self.of_list, res_opt)

def return_noise_arrays(sigma_xy, sigma_r, x_pos):
    
    # The arrays are of size (# of holes, 4) because in each for loop for design_noisy_phc, atmost 1 hole is placed in each quadrant
    # Exceptions are the holes in x,y axis but as long as I use the same code to define the PhC everywhere, things should be consistent
    r_noise_array = np.random.normal(loc = 0, scale = sigma_r, size = (len(x_pos), 4))
    x_noise_array = np.random.normal(loc = 0, scale = sigma_xy, size = (len(x_pos), 4))
    y_noise_array = np.random.normal(loc = 0, scale = sigma_xy, size = (len(x_pos), 4))
    
    # Append everyhting in one array
    noise_arrays = [x_noise_array, y_noise_array, r_noise_array]
    
    return noise_arrays

def design_noisy_phc(Nx, Ny, x_pos, y_pos, hole_radius, slab_thickness, refractive_index, dx ,dy, noise_arrays):
    
    '''Initialise a L3 photonic crystal with added gaussian noise to (hole position, diameter or both)'''

    # Initialize a lattice and PhC
    lattice = legume.Lattice([Nx, 0], [0, Ny*np.sqrt(3)/2])
    L3_phc = legume.PhotCryst(lattice)

    # Add a layer to the PhC
    L3_phc.add_layer(d=slab_thickness, eps_b=refractive_index**2)
    
    # Get the position of holes for L3 cavity
    nx, ny = Nx//2 + 1, Ny//2 + 1

    for ic, x in enumerate(x_pos):
        
        x_noise = noise_arrays[0][ic]
        y_noise = noise_arrays[1][ic]
        r_noise = noise_arrays[2][ic]
        
        yc = y_pos[ic] if y_pos[ic] == 0 else y_pos[ic] + dy[ic]
        xc = x if x == 0 else x_pos[ic] + dx[ic]
        L3_phc.add_shape(legume.Circle(x_cent=xc + x_noise[0], y_cent=yc + y_noise[0], r=hole_radius + r_noise[0]))
        if nx-0.6 > x_pos[ic] > 0 and (ny-1.1)*np.sqrt(3)/2 > y_pos[ic] > 0:
            L3_phc.add_shape(legume.Circle(x_cent=-xc + x_noise[1], y_cent=-yc + y_noise[1], r=hole_radius + r_noise[1]))
        if nx-1.6 > x_pos[ic] > 0:
            L3_phc.add_shape(legume.Circle(x_cent=-xc + x_noise[2], y_cent=yc + y_noise[2], r=hole_radius + r_noise[2]))
        if (ny-1.1)*np.sqrt(3)/2 > y_pos[ic] > 0 and nx-1.1 > x_pos[ic]:
            L3_phc.add_shape(legume.Circle(x_cent=xc + x_noise[3], y_cent=-yc + y_noise[3], r=hole_radius + r_noise[3]))
            
    return L3_phc
    
def define_noisy_gme(Nx, Ny, x_pos, y_pos, hole_radius, slab_thickness, refractive_index, dx, dy, sigma_xy, sigma_r,
                     gaussian_width, eigvec, gmax, kpoints_array, gme_options):
    
    noise_arrays = return_noise_arrays(sigma_r, sigma_xy, x_pos)
    
    noisy_L3_phc = design_noisy_phc(Nx, Ny, x_pos, y_pos, hole_radius, slab_thickness, 
                                    refractive_index, dx, dy, noise_arrays)
    
    # L3_phc = design_phc(Nx, Ny, x_pos, y_pos, hole_radius, slab_thickness, refractive_index, dx, dy)
  
    (gme, f_imag_array, direction_ratio_array, indmode) = gme_cavity_dot(noisy_L3_phc, gaussian_width, eigvec, gmax, kpoints = kpoints_array, gme_options = gme_options)
    
    quality_factor = calculate_Q(gme.freqs, f_imag_array, indmode)
    direction_ratio = np.mean(direction_ratio_array)
    
    return quality_factor, direction_ratio

def run_noisy_gme(Nx, Ny, x_pos, y_pos, hole_radius, slab_thickness, 
                        refractive_index, dx, dy, sigma_xy, sigma_r,
                        gaussian_width, eigvec, gmax, kpoints_array, gme_options, randomize_runs):

    results = Parallel(n_jobs=3, backend="threading")(delayed(define_noisy_gme)(Nx, Ny, x_pos, y_pos, hole_radius, slab_thickness, 
                                                                                 refractive_index, dx, dy, sigma_xy, sigma_r,
                                                                                 gaussian_width, eigvec, gmax, kpoints_array, gme_options)
                                                                                for _ in range(randomize_runs))
    quality_factor = np.zeros(randomize_runs)
    direction_ratio = np.zeros(randomize_runs)
    
    for i in range(randomize_runs):
        quality_factor[i] = results[i][0]
        direction_ratio[i] = results[i][1]
    
    return quality_factor, direction_ratio

def noisy_loss_function(quality_factor, direction_ratio, Q_target):
    
    if npa.mean(quality_factor) < 0.7*10**(Q_target-1):
        loss_function = npa.mean(
            (npa.pi/4 - npa.arctan(quality_factor/10**(Q_target-1))**2 + (1 - direction_ratio)**2))
    else:
        loss_function = npa.mean(
            (npa.pi/4 - npa.arctan(quality_factor/10**(Q_target))**2 + (1 - direction_ratio)**2))
        
    return loss_function

def return_field_overlap(gme, kind, indmode, x_pos, y_pos):
    
    E = gme.get_field_xy('E', kind = kind, mind = indmode, z = 0)
    E_xy = (np.abs(E[0]['x'])**2 + np.abs(E[0]['y'])**2)/np.max((np.abs(E[0]['x'])**2 + np.abs(E[0]['y'])**2))
    
    x = np.linspace(-max(x_pos), max(x_pos), 100)
    y = np.linspace(-max(y_pos), max(y_pos), 100)
    
    field_hole_overlap = 0
    for (idx, x_val) in enumerate(x_pos):
        y_val = y_pos[idx]
        x_idx = np.argmin(np.abs(x_val - x))
        y_idx = np.argmin(np.abs(y_val - y))
        field_hole_overlap = E_xy[x_idx, y_idx] + field_hole_overlap
        
    return field_hole_overlap

def save_eigvecs(gme, indmode, filename):
    # Save the eigenvectors corresponding to the cavity mode
    # This method is used to track the mode when doing inverse design
    
    eigvec = gme.eigvecs[0][:, indmode]
    np.save(filename, eigvec, allow_pickle=True)
    
def plotting_with_weights(weights_name):
    
    weights = np.load(weights_name, allow_pickle=True)
    
    (size, ) = weights.shape
    
    param_history = weights[0]

    loss_function = weights[1]
    t_elapsed = weights[2]
    
    # Directionality and Quality factor are in thruples
    directionality = weights[3]
    quality_factor = weights[4]
    
    epochs = np.arange(len(loss_function)) + 1
    
    fig = plt.figure(figsize=(15, 8))
    spec = fig.add_gridspec(ncols=1, nrows=size-1) # subplot grid
        
    plt.rcParams.update({'font.size': 14})  # Change default font size for text

    fig.add_subplot(spec[0, 0])
    plt.plot(epochs, quality_factor)
    plt.yscale('log')
    plt.grid(True)
    plt.ylabel('Quality Factor')
    
    fig.add_subplot(spec[1, 0])
    plt.plot(epochs, directionality)
    plt.ylabel('Directionality')
    plt.grid(True)
    plt.yscale('log')
    plt.ylim([10**int(np.log10(np.min(directionality))-1), 2])
    
    fig.add_subplot(spec[2, 0])
    plt.plot(epochs, t_elapsed)
    plt.ylabel('time/epoch (s)')
    plt.xticks([])
    plt.grid(True)
    
    fig.add_subplot(spec[3, 0])
    plt.plot(epochs, loss_function)
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.ylim([10**int(np.log10(np.min(loss_function))-2), 5])
    plt.xlabel('Epochs')
        
    head, tail = os.path.split(weights_name)
    plt.suptitle(tail[:-4])

    fig_save_file = f'./{head}/plots/' + tail[:-4] + '.png'
    fig.savefig(fig_save_file)
    
    param_save_file = f'./{head}/param_history/' + tail[:-4] + '_param_history.npy'
    np.save(param_save_file, param_history)
    
    plt.show()    