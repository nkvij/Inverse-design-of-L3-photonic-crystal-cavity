import numpy as np
import matplotlib.pyplot as plt
import os
import tidy3d as td
from tidy3d.plugins.resonance import ResonanceFinder
from tidy3d import web

import gdstk

def convert_freq_to_wavelength(freq):
    """Convert frequency to wavelength (nm)"""
    
    wavelength = (td.C_0/freq)
    return wavelength

def L3_params_930():
    '''
    The thickness of slab is incorrect: 180 nm as compared to the real thickness: 190 nm
    '''
    # Periodicity in units of PhC periods in x and y directions
    Nx, Ny = 16, 16
    
    # Lattice constant of the PhC in micron
    alattice = 0.23

    hole_radius = 0.07
    # slab_thickness = 0.782*alattice
    slab_thickness = 0.19
    refractive_index = 3.55
    x_shift = 0.042
    
    return Nx, Ny, alattice, hole_radius, slab_thickness, refractive_index, x_shift

def L3_params_936():
    
    # Periodicity in units of PhC periods in x and y directions
    Nx, Ny = 16, 16
    
    # Lattice constant of the PhC in micron
    alattice = 0.23

    # Regular PhC lattice parameterssss
    hole_radius = 0.072  # radius of holes in regular PhC region
    slab_thickness = 0.19 # slab thickness
    refractive_index = 3.55  # refractive index of the slab
    x_shift = 0.041
    
    return Nx, Ny, alattice, hole_radius, slab_thickness, refractive_index, x_shift

def L3_params_945():
    
    # Periodicity in units of PhC periods in x and y directions
    Nx, Ny = 16, 16
    
    # Lattice constant of the PhC in micron
    alattice = 0.23

    # Regular PhC lattice parameters
    hole_radius = 0.07  # radius of holes in regular PhC region
    slab_thickness = 0.19 # slab thickness
    refractive_index = 3.55  # refractive index of the slab
    x_shift = 0.042
    
    return Nx, Ny, alattice, hole_radius, slab_thickness, refractive_index, x_shift

def L3_params_520():
    
    # Periodicity in units of PhC periods in x and y directions
    Nx, Ny = 16, 16
    
    # Lattice constant of the PhC in micron
    alattice = 0.208

    # Regular PhC lattice parameters
    hole_radius = 0.3*alattice  # radius of holes in regular PhC region
    slab_thickness = 0.225 # slab thickness
    refractive_index = 2  # refractive index of the slab
    x_shift = 0.26*alattice
    
    return Nx, Ny, alattice, hole_radius, slab_thickness, refractive_index, x_shift

def L3_params_440():
    
    # Periodicity in units of PhC periods in x and y directions
    Nx, Ny = 16, 16
    
    # Lattice constant of the PhC in micron
    alattice = 0.145

    # Regular PhC lattice parameters
    hole_radius = 0.3*alattice  # radius of holes in regular PhC region
    slab_thickness = 0.085 # slab thickness
    refractive_index = 2.75  # refractive index of the slab
    x_shift = 0.1*alattice
    
    return Nx, Ny, alattice, hole_radius, slab_thickness, refractive_index, x_shift

def L3_params_780():
    
    # Periodicity in units of PhC periods in x and y directions
    Nx, Ny = 16, 16
    
    # Lattice constant of the PhC in micron
    alattice = 0.200

    # Regular PhC lattice parameters
    hole_radius = 0.3*alattice  # radius of holes in regular PhC region
    slab_thickness = 0.140 # slab thickness
    refractive_index = 3.38  # refractive index of the slab
    x_shift = 0.2*alattice
    
    return Nx, Ny, alattice, hole_radius, slab_thickness, refractive_index, x_shift

def L3_params_1550():
    
    # Periodicity in units of PhC periods in x and y directions
    Nx, Ny = 16, 16
    
    # Lattice constant of the PhC in micron
    alattice = 0.4

    # Regular PhC lattice parameters
    hole_radius = 0.1  # radius of holes in regular PhC region
    slab_thickness = 0.22 # slab thickness
    refractive_index = 3.4757  # refractive index of the slab
    x_shift = 0
    
    return Nx, Ny, alattice, hole_radius, slab_thickness, refractive_index, x_shift

def return_dipole_source(freq0, fwidth, center_location, polarization, phase = 0):
    """Return dipole source emitting a gaussian pulse with center frequency freq0 and bandwidth fwidth

    Args:
        freq0 (float): center frequency of dipole (Hz)
        fwidth (float): bandwidth of dipole source (Hz)
        center_location (1x3 np array): coorinates of the center of the dipole source (um)
        polarization (string): polarization direction of the dipole source
        phase (float): relative phase of the gaussian pulse of the dipole source

    Returns:
        'PointDipole': tidy3D object of dipole
    """
    # Source; plot time dependence to verify when the source pulse decayed
    source = td.PointDipole(
        center=center_location,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth, phase = phase),
        polarization=polarization,
    )

    return source

def return_gaussian_input_pulse(freq0, fwidth, center, size, waist_radius):
    
    # Source bandwidth.
    pulse = td.GaussianPulse(freq0=freq0, fwidth=fwidth)

    # Source definition
    gauss_source = td.GaussianBeam(
    center=center,
    size=size,
    source_time=pulse,
    direction='-',
    pol_angle=np.pi/2,
    angle_theta=0,
    angle_phi=0,
    waist_radius=waist_radius,
    waist_distance = center[2],
    name="gauss_source",
    )

    return gauss_source

def define_mesh(alattice, steps_per_unit_length):
    # Suppress warnings for some of the holes being too close to the PML
    td.config.logging_level = "ERROR"

    # Mesh step in x, y, z, in micron
    grid_spec = td.GridSpec(
        grid_x=td.UniformGrid(dl=alattice / steps_per_unit_length),
        grid_y=td.UniformGrid(dl=alattice / steps_per_unit_length * np.sqrt(3) / 2),
        grid_z=td.AutoGrid(min_steps_per_wvl=steps_per_unit_length),
    )

    return grid_spec

def get_time_series_monitor(t_start = 0.5e-13, center = [0, 0, 0], size = [0, 0, 0], name = "time_series"):
    
    # Time series monitor for Q-factor computation
    time_series_monitor = td.FieldTimeMonitor(center=center, size=size, start=t_start, name=name)
    
    return time_series_monitor

def get_near_field_monitor(freq_range, t_start = 0.5e-13, width = 2e-14, center = [0, 0, 0], size = [4, 2*np.sqrt(3), 0], name = "field"):
    
    # Apodization to exclude the source pulse from the frequency-domain monitors
    apodization = td.ApodizationSpec(start=t_start, width=width)
    
    # near field
    near_field_monitor = td.FieldMonitor(
        center = center,
        size = size,
        freqs = freq_range,
        name = name,
        apodization = apodization,
    )
    
    return near_field_monitor

def get_far_field_monitor(slab_thickness, freq_range, t_start = 0.5e-13, width = 2e-14, name = "far_field monitor"):
    
    # far field
    ux = np.linspace(-1, 1, 101)
    uy = np.linspace(-1, 1, 101)
    
    # Apodization to exclude the source pulse from the frequency-domain monitors
    apodization = td.ApodizationSpec(start=t_start, width=width)
    
    far_field_monitor = td.FieldProjectionKSpaceMonitor(
        center=(0, 0, slab_thickness/2 + 0.1),
        size=(td.inf, td.inf, 0),
        freqs=freq_range,
        name=name,
        proj_axis=2,
        ux=ux,
        uy=uy,
        apodization=apodization
    )
    
    return far_field_monitor

def get_flux_monitor(sim_size, freq_range):
    
    # Flux monitor to get the total dipole power.
    flux_monitor = td.FluxMonitor(
        center=(0, 0, sim_size[2]/2),
        size=(sim_size[0], sim_size[1], sim_size[2]),
        freqs=freq_range,
        name="flux_dip",
    )

    return flux_monitor

def calculate_purcell_factor(resonant_wavelength, refractive_index, quality_factor, mode_volume):
    
    # p_bulk_a = ((2 * np.pi * freq_range) ** 2 / (12 * np.pi)) * (td.MU_0 * refractive_index / td.C_0)
    # p_bulk_a = p_bulk_a * 2 ** (2 * np.sum(np.abs(simulation.symmetry)))
    
    # p_cav = sim_data["flux_dip"].flux
    
    # purell_factor = p_cav / p_bulk_a
    
    purcell_factor = (3 / (4 * np.pi**2)) * ((resonant_wavelength / refractive_index) ** 3) * (quality_factor / mode_volume)
    
    return purcell_factor

def phc_holes_array(Nx, Ny, alattice, cavity_type):
    
    x_pos, y_pos = [], []
    nx, ny = Nx//2 + 1, Ny//2 + 1
    
    if(cavity_type == 'L3' or cavity_type == 'l3'):
        for iy in range(ny):
            for ix in range(nx):
                if(ix == 0 and iy == 0 or ix == 1 and iy == 0): # skip holes for L3 cavity
                    continue
                else:
                    x_pos.append((ix + (iy%2)*0.5)*alattice)
                    y_pos.append((iy*np.sqrt(3)/2)*alattice)
    
    if(cavity_type == 'H1' or cavity_type == 'h1'):
        for iy in range(ny):
            for ix in range(nx):
                if(ix == 0 and iy == 0): # skip holes for H1 cavity
                    continue
                else:
                    x_pos.append((ix + (iy%2)*0.5)*alattice)
                    y_pos.append((iy*np.sqrt(3)/2)*alattice)
    
    return x_pos, y_pos

def get_phc(Nx, Ny, x_pos, y_pos, alattice, hole_radius, slab_thickness, refractive_index, dx ,dy):
    """Get the hole geometry for PhC"""
    
    # Initialize structures
    slab = td.Structure(geometry=td.Box(center=[0, 0, 0], size=[td.inf, td.inf, slab_thickness]), medium= refractive_index)
    
    holes_group = []
    
    # Get the position of holes for L3 cavity
    nx, ny = Nx//2 + 1, Ny//2 + 1

    # Apply holes symmetrically in the four quadrants
    for ic, x in enumerate(x_pos):
        yc = y_pos[ic] if y_pos[ic] == 0 else y_pos[ic] + dy[ic]
        xc = x if x == 0 else x_pos[ic] + dx[ic]
        
        # Add gaussian noise to the hole raduis
        hole_radius = hole_radius
        
        holes_group.append(td.Cylinder(center=[xc, yc, 0], radius=hole_radius, length=slab_thickness))      
        if nx-alattice/2+0.1 > x_pos[ic] > 0 and (ny-alattice+0.1)*np.sqrt(3)/2 > y_pos[ic] > 0:
            holes_group.append(td.Cylinder(center=[-xc, -yc, 0], radius=hole_radius, length=slab_thickness))      
        if nx-1.5*alattice+0.1 > x_pos[ic] > 0:
            holes_group.append(td.Cylinder(center=[-xc, yc, 0], radius=hole_radius, length=slab_thickness))      
        if (ny-alattice+0.1)*np.sqrt(3)/2 > y_pos[ic] > 0 and nx-alattice+0.1 > x_pos[ic]:
            holes_group.append(td.Cylinder(center=[xc, -yc, 0], radius=hole_radius, length=slab_thickness))     

    holes = td.Structure(geometry=td.GeometryGroup(geometries=holes_group), medium=td.Medium())

    return slab, holes

def return_noise_arrays(sigma_xy, sigma_r, x_pos):
    
    # The arrays are of size (# of holes, 4) because in each for loop for design_noisy_phc, atmost 1 hole is placed in each quadrant
    # Exceptions are the holes in x,y axis but as long as I use the same code to define the PhC everywhere, things should be consistent
    r_noise_array = np.random.normal(loc = 0, scale = sigma_r, size = (len(x_pos), 4))
    x_noise_array = np.random.normal(loc = 0, scale = sigma_xy, size = (len(x_pos), 4))
    y_noise_array = np.random.normal(loc = 0, scale = sigma_xy, size = (len(x_pos), 4))
    
    # Append everyhting in one array
    noise_arrays = [x_noise_array, y_noise_array, r_noise_array]
    
    return noise_arrays

def get_noisy_phc(Nx, Ny, x_pos, y_pos, alattice, hole_radius, slab_thickness, refractive_index, dx ,dy, noise_arrays):
    """Get the hole geometry for PhC"""
    
    # Initialize structures
    slab = td.Structure(geometry=td.Box(center=[0, 0, 0], size=[td.inf, td.inf, slab_thickness]), medium= refractive_index)
    
    holes_group = []
    
    # Get the position of holes for L3 cavity
    nx, ny = Nx//2 + 1, Ny//2 + 1

    for ic, x in enumerate(x_pos):
        yc = y_pos[ic] if y_pos[ic] == 0 else y_pos[ic] + dy[ic]
        xc = x if x == 0 else x_pos[ic] + dx[ic]
        
        x_noise = noise_arrays[0][ic]
        y_noise = noise_arrays[1][ic]
        r_noise = noise_arrays[2][ic]

        holes_group.append(td.Cylinder(center=[xc + x_noise[0], yc + y_noise[0], 0], radius=hole_radius + r_noise[0], length=slab_thickness))      
        if nx-alattice/2+0.1 > x_pos[ic] > 0 and (ny-alattice+0.1)*np.sqrt(3)/2 > y_pos[ic] > 0:
            holes_group.append(td.Cylinder(center=[-xc + x_noise[1], -yc + y_noise[1], 0], radius=hole_radius + r_noise[1], length=slab_thickness))      
        if nx-1.5*alattice+0.1 > x_pos[ic] > 0:
            holes_group.append(td.Cylinder(center=[-xc + x_noise[2], yc + y_noise[2], 0], radius=hole_radius + r_noise[2], length=slab_thickness))      
        if (ny-alattice+0.1)*np.sqrt(3)/2 > y_pos[ic] > 0 and nx-alattice+0.1 > x_pos[ic]:
            holes_group.append(td.Cylinder(center=[xc + x_noise[3], -yc + y_noise[3], 0], radius=hole_radius + r_noise[3], length=slab_thickness))     

    
    holes = td.Structure(geometry=td.GeometryGroup(geometries=holes_group), medium=td.Medium())

    return slab, holes 
                  
def get_H1_cavity(Nx, Ny, alattice, r_hole, d_slab):
    """Get the hole geometry for the L3 cavity"""
    
    # Initialize structures
    slab = td.Structure(
        geometry=td.Box(center=[0, 0, 0], size=[td.inf, td.inf, d_slab]), medium= gaas
    )
    
    holes_group = []
    
    for y in range(int(Nx/2)):
        for x in range(int(Ny/2)):
    
            if(x == 0 and y == 0): # skip holes for L3 cavity
                continue
                
            elif(x == 2 and y == 0):
                holes_group.append(td.Cylinder(center=[x*alattice + shift, y*alattice*np.sqrt(3)/2, 0], radius=r_hole, length=d_slab))      
                holes_group.append(td.Cylinder(center=[-x*alattice - shift, y*alattice*np.sqrt(3)/2, 0], radius=r_hole, length=d_slab))
                
            elif(y%2 == 0):
                holes_group.append(td.Cylinder(center=[x*alattice, y*alattice*np.sqrt(3)/2, 0], radius=r_hole, length=d_slab))      
                holes_group.append(td.Cylinder(center=[-x*alattice, y*alattice*np.sqrt(3)/2, 0], radius=r_hole, length=d_slab))      
                holes_group.append(td.Cylinder(center=[x*alattice, -y*alattice*np.sqrt(3)/2, 0], radius=r_hole, length=d_slab))      
                holes_group.append(td.Cylinder(center=[-x*alattice, -y*alattice*np.sqrt(3)/2, 0], radius=r_hole, length=d_slab))
                
            elif(y%2 == 1):
                holes_group.append(td.Cylinder(center=[(x + 1/2)*alattice, y*alattice*np.sqrt(3)/2, 0], radius=r_hole, length=d_slab))      
                holes_group.append(td.Cylinder(center=[-(x + 1/2)*alattice, y*alattice*np.sqrt(3)/2, 0], radius=r_hole, length=d_slab))      
                holes_group.append(td.Cylinder(center=[(x + 1/2)*alattice, -y*alattice*np.sqrt(3)/2, 0], radius=r_hole, length=d_slab))      
                holes_group.append(td.Cylinder(center=[-(x + 1/2)*alattice, -y*alattice*np.sqrt(3)/2, 0], radius=r_hole, length=d_slab))
                
    holes = td.Structure(geometry=td.GeometryGroup(geometries=holes_group), medium=air)
    return slab, holes

def convert_phc_size(Nx_orig, Ny_orig, Nx_final, Ny_final, alattice, dx, dy):
    
    nx_orig, ny_orig = Nx_orig//2 + 1, Ny_orig//2 + 1
    nx_final, ny_final = Nx_final//2 + 1, Ny_final//2 + 1
    
    x_pos_new = []
    y_pos_new = []
    
    for iy in range(ny_final):
            for ix in range(nx_final):
                if(ix == 0 and iy == 0 or ix == 1 and iy == 0): # skip holes for L3 cavity
                    continue
                else:
                    x_pos_new.append((ix + (iy%2)*0.5)*alattice)
                    y_pos_new.append((iy*np.sqrt(3)/2)*alattice)
    
    dx_new = np.zeros(len(x_pos_new))
    dy_new = np.zeros(len(y_pos_new))
        
    dx_temp = np.zeros(nx_orig*ny_orig)
    dy_temp = np.zeros(nx_orig*ny_orig)
    
    dx_temp[2:] = dx
    dy_temp[2:] = dy
    
    dx_temp = np.reshape(dx_temp, (nx_orig, ny_orig))
    dy_temp = np.reshape(dy_temp, (nx_orig, ny_orig))
    
    dx_temp = np.pad(dx_temp, (0, nx_final - nx_orig))
    dy_temp = np.pad(dy_temp, (0, ny_final - ny_orig))
    
    dx_temp = np.reshape(dx_temp, (1, nx_final*ny_final))
    dy_temp = np.reshape(dy_temp, (1, nx_final*ny_final))
    
    dx_new = dx_temp[0, 2:]
    dy_new = dy_temp[0, 2:]
    
    return x_pos_new, y_pos_new, dx_new, dy_new
    
def analyse_time_monitor_data(sim_data, freq_range, freq_window, plot_bool = True, print_data = False):
      # Get data from the TimeMonitor
    tdata = sim_data["time_series"]

    time_series = tdata.Ey.squeeze()

    if(plot_bool):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

        # Plot time dependence
        time_series.plot(ax=ax1)

        # Make frequency mesh and plot spectrum
        dt = sim_data.simulation.dt
        fmesh = np.linspace(-1 / dt / 2, 1 / dt / 2, time_series.size)
        spectrum = np.fft.fftshift(np.fft.fft(time_series))

        ax2.plot(fmesh, np.abs(spectrum))
        ax2.set_xlim(freq_window)
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("Electric field [a.u.]")
        ax2.set_title("Spectrum")
        plt.show()

    resonance_finder = ResonanceFinder(freq_window=freq_window, init_num_freqs=500)
    resonance_data = resonance_finder.run(sim_data["time_series"])
    
    if(print_data):
        print(resonance_data.to_dataframe())

    # resonance_data = resonance_data.where(resonance_data.freq < freq_range[0], drop=True)
    
    freq_index = np.argmin(np.asarray(resonance_data.error))
    resonant_frequency = np.asarray(resonance_data.freq)[freq_index]
    index = np.argmin(np.abs(freq_range - resonant_frequency))
    quality_factor = np.asarray(resonance_data.Q)[freq_index]
    resonant_wavelength = np.round(convert_freq_to_wavelength(resonant_frequency)*1000,2)

    return index, quality_factor, resonant_wavelength

def build_layers(n_1, n_2, N, z_0, lda0):
    # n_1 and n_2 are the refractive indices of the two materials
    # N is the number of repeated pairs of low/high refractive index material
    # z_0 is the z coordinate of the lowest layer
    # lda0 is the central wavelength
    
    freq0 = td.C_0 / lda0  # central frequency
    inf_eff = 10  # effective infinity in this model


    material_1 = td.Medium(permittivity=n_1**2)  # define the first material
    material_2 = td.Medium(permittivity=n_2**2)  # define the second material
    t_1 = lda0 / (4 * n_1)  # thickness of the first material
    t_2 = lda0 / (4 * n_2)  # thicness of the second material
    layers = []  # holder for all the layers

    # building layers alternatively
    for i in range(2 * N):
        if i % 2 == 0:
            layers.append(
                td.Structure(
                    geometry=td.Box.from_bounds(
                        rmin=(-inf_eff, -inf_eff, z_0),
                        rmax=(inf_eff, inf_eff, z_0 + t_1),
                    ),
                    medium=material_1,
                )
            )
            z_0 = z_0 + t_1
        else:
            layers.append(
                td.Structure(
                    geometry=td.Box.from_bounds(
                        rmin=(-inf_eff, -inf_eff, z_0),
                        rmax=(inf_eff, inf_eff, z_0 + t_2),
                    ),
                    medium=material_2,
                )
            )
            z_0 = z_0 + t_2

    # DBR = td.Structure(geometry=td.Box(geometries=layers))
    return layers

def return_reciprocal_vectors_936nm():
    
    y_gvec = np.asarray([-0.90689968, -0.45344984,  0.        ,  0.45344984,  0.90689968,
       -0.90689968, -0.45344984,  0.        ,  0.45344984,  0.90689968,
       -1.36034952, -0.90689968, -0.45344984,  0.        ,  0.45344984,
        0.90689968,  1.36034952, -1.36034952, -0.90689968, -0.45344984,
        0.        ,  0.45344984,  0.90689968,  1.36034952, -1.36034952,
       -0.90689968, -0.45344984,  0.        ,  0.45344984,  0.90689968,
        1.36034952, -0.90689968, -0.45344984,  0.        ,  0.45344984,
        0.90689968, -0.90689968, -0.45344984,  0.        ,  0.45344984,
        0.90689968])
    x_gvec = np.asarray([-1.17809725, -1.17809725, -1.17809725, -1.17809725, -1.17809725,
        -0.78539816, -0.78539816, -0.78539816, -0.78539816, -0.78539816,
        -0.39269908, -0.39269908, -0.39269908, -0.39269908, -0.39269908,
        -0.39269908, -0.39269908,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.39269908,
            0.39269908,  0.39269908,  0.39269908,  0.39269908,  0.39269908,
            0.39269908,  0.78539816,  0.78539816,  0.78539816,  0.78539816,
            0.78539816,  1.17809725,  1.17809725,  1.17809725,  1.17809725,
            1.17809725])

    # x_gvec_N_10 = np.asarray([-1.25663706, -1.25663706, -1.25663706, -0.62831853, -0.62831853,
    #     -0.62831853,  0.        ,  0.        ,  0.        ,  0.        ,
    #         0.        ,  0.62831853,  0.62831853,  0.62831853,  1.25663706,
    #         1.25663706,  1.25663706])
    # y_gvec_N_10 = np.asarray([-0.72551975,  0.        ,  0.72551975, -0.72551975,  0.        ,
    #         0.72551975, -1.45103949, -0.72551975,  0.        ,  0.72551975,
    #         1.45103949, -0.72551975,  0.        ,  0.72551975, -0.72551975,
    #         0.        ,  0.72551975])

    k_max = np.sqrt(np.max(x_gvec**2 + y_gvec**2))

    # x_gvec = x_gvec_N_10/k_max
    # y_gvec = y_gvec_N_10/k_max

    y_gvec = y_gvec/k_max
    x_gvec = x_gvec/k_max
    
    return x_gvec, y_gvec

def return_reciprocal_vectors_520nm():
    
    y_gvec = np.asarray([-0.45344984,  0.        ,  0.45344984, -1.36034952, -0.90689968,
       -0.45344984,  0.        ,  0.45344984,  0.90689968,  1.36034952,
       -1.81379936, -1.36034952, -0.90689968, -0.45344984,  0.        ,
        0.45344984,  0.90689968,  1.36034952,  1.81379936, -1.81379936,
       -1.36034952, -0.90689968, -0.45344984,  0.        ,  0.45344984,
        0.90689968,  1.36034952,  1.81379936, -2.26724921, -1.81379936,
       -1.36034952, -0.90689968, -0.45344984,  0.        ,  0.45344984,
        0.90689968,  1.36034952,  1.81379936,  2.26724921, -2.26724921,
       -1.81379936, -1.36034952, -0.90689968, -0.45344984,  0.        ,
        0.45344984,  0.90689968,  1.36034952,  1.81379936,  2.26724921,
       -2.26724921, -1.81379936, -1.36034952, -0.90689968, -0.45344984,
        0.        ,  0.45344984,  0.90689968,  1.36034952,  1.81379936,
        2.26724921, -2.26724921, -1.81379936, -1.36034952, -0.90689968,
       -0.45344984,  0.        ,  0.45344984,  0.90689968,  1.36034952,
        1.81379936,  2.26724921, -2.26724921, -1.81379936, -1.36034952,
       -0.90689968, -0.45344984,  0.        ,  0.45344984,  0.90689968,
        1.36034952,  1.81379936,  2.26724921, -1.81379936, -1.36034952,
       -0.90689968, -0.45344984,  0.        ,  0.45344984,  0.90689968,
        1.36034952,  1.81379936, -1.81379936, -1.36034952, -0.90689968,
       -0.45344984,  0.        ,  0.45344984,  0.90689968,  1.36034952,
        1.81379936, -1.36034952, -0.90689968, -0.45344984,  0.        ,
        0.45344984,  0.90689968,  1.36034952, -0.45344984,  0.        ,
        0.45344984])
    
    x_gvec = np.asarray([-2.35619449, -2.35619449, -2.35619449, -1.96349541, -1.96349541,
       -1.96349541, -1.96349541, -1.96349541, -1.96349541, -1.96349541,
       -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.57079633,
       -1.57079633, -1.57079633, -1.57079633, -1.57079633, -1.17809725,
       -1.17809725, -1.17809725, -1.17809725, -1.17809725, -1.17809725,
       -1.17809725, -1.17809725, -1.17809725, -0.78539816, -0.78539816,
       -0.78539816, -0.78539816, -0.78539816, -0.78539816, -0.78539816,
       -0.78539816, -0.78539816, -0.78539816, -0.78539816, -0.39269908,
       -0.39269908, -0.39269908, -0.39269908, -0.39269908, -0.39269908,
       -0.39269908, -0.39269908, -0.39269908, -0.39269908, -0.39269908,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.39269908,  0.39269908,  0.39269908,  0.39269908,
        0.39269908,  0.39269908,  0.39269908,  0.39269908,  0.39269908,
        0.39269908,  0.39269908,  0.78539816,  0.78539816,  0.78539816,
        0.78539816,  0.78539816,  0.78539816,  0.78539816,  0.78539816,
        0.78539816,  0.78539816,  0.78539816,  1.17809725,  1.17809725,
        1.17809725,  1.17809725,  1.17809725,  1.17809725,  1.17809725,
        1.17809725,  1.17809725,  1.57079633,  1.57079633,  1.57079633,
        1.57079633,  1.57079633,  1.57079633,  1.57079633,  1.57079633,
        1.57079633,  1.96349541,  1.96349541,  1.96349541,  1.96349541,
        1.96349541,  1.96349541,  1.96349541,  2.35619449,  2.35619449,
        2.35619449])
    
    k_max = np.sqrt(np.max(x_gvec**2 + y_gvec**2))

    # x_gvec = x_gvec_N_10/k_max
    # y_gvec = y_gvec_N_10/k_max

    y_gvec = y_gvec/k_max
    x_gvec = x_gvec/k_max
    
    return x_gvec, y_gvec


    
    if(ff_type == 'gaussian'):
        x = np.linspace(-1, 1, 101)
        y = np.linspace(-1, 1, 101)
        dx = x[1] - x[0]  # spacing along x-axis

        x_, y_ = np.meshgrid(x, y)

        width = np.linspace(0, 20, 500)
        overlap_integral_array = np.zeros(len(width))

        power = np.sqrt(np.nan_to_num(far_field))
        norm1 = np.sqrt(np.trapz(np.trapz(power**2, dx=dx),dx=dx))
        power = power/norm1

        for i in range(len(width)):
            gaussian = np.exp(-(x_**2 + y_**2)*width[i])
            norm2 = np.sqrt(np.trapz(np.trapz(gaussian**2, dx=dx),dx=dx))
            gaussian = gaussian/norm2

            integrated_value = np.trapz(np.trapz(power * gaussian, dx = dx), dx=dx)
            overlap_integral_array[i] = integrated_value**2
            width_max = width[np.argmax(overlap_integral_array)]

        if(bool_plot):
            plt.figure()
            plt.plot(width, overlap_integral_array)
            plt.grid(True)
            plt.title(f"Max overlap = {np.round(np.max(overlap_integral_array)*100,2)}%, width = {np.round(width_max,2)}")
            plt.axvline(x = width_max, color = 'k', linestyle = '--')
            plt.xlabel("Gaussian width (arb. units)")
            plt.ylabel("Overlap")
            plt.show()

            gaussian = np.exp(-(x_**2 + y_**2)*width_max)
            norm2 = np.sqrt(np.trapz(np.trapz(gaussian**2, axis=0, dx=dx), axis=0,dx=dx))
            gaussian = gaussian/norm2

            fig, (ax1, ax2) = plt.subplots(1,2)
            im = ax1.imshow(gaussian, extent=[x.min(), x.max(), y.min(), y.max()], vmax=np.max(power), vmin=0)
            im = ax2.imshow(power, extent=[x.min(), x.max(), y.min(), y.max()], vmax=np.max(power), vmin=0)

            phis = np.linspace(0, np.pi * 2, 201)
            NA = 1/width_max
            ax1.plot(NA * np.cos(phis), NA * np.sin(phis), lw=2, color="w", linestyle = "--")
            ax2.plot(NA * np.cos(phis), NA * np.sin(phis), lw=2, color="w", linestyle = "--")
            ax2.yaxis.set_ticklabels([])
            plt.show()

            plt.figure()
            plt.plot(x, power[:,101//2])
            plt.plot(x, gaussian[:,101//2])

        return np.max(overlap_integral_array)
    
    if(ff_type == 'directional'):
        
        eta = calculate_emission_into_NA(far_field, NA, far_field.ux, far_field.uy)
        return eta
    
def calculate_emission_into_NA(far_field_E, NA, ux, uy):
    
    x_, y_ = np.meshgrid(ux, uy)
    sine_theta = np.sqrt(x_**2 + y_**2)
    
    far_field_E = np.nan_to_num(far_field_E)
    integral_norm = np.trapz(np.trapz(np.nan_to_num(far_field_E)))

    E_circle = np.where(sine_theta < NA, far_field_E/integral_norm, 0)
    eta = np.trapz(np.trapz((E_circle), axis = -1))

    return eta

def calculate_directionality(far_field, ff_type, NA, bool_plot = True):
    
    if(ff_type == 'gaussian'):
        x = np.linspace(-0.99, 0.99, 101)
        y = np.linspace(-0.99, 0.99, 101)

        x_, y_ = np.meshgrid(x, y)
        x_ = 4*x_/np.sqrt(1 - x_**2)
        y_ = 4*y_/np.sqrt(1 - y_**2)
        dx = x[1] - x[0]  # spacing along x-axis

        width = np.linspace(0.1, 5, 500)
        overlap_integral_array = np.zeros(len(width))

        # Integral of E**2 for normalization
        # print(float(far_field.max()))
        far_field = np.nan_to_num(far_field)/float(far_field.max())
        far_field_norm = np.trapz(np.trapz(far_field**2, dx=dx),dx=dx)

        for i in range(len(width)):
            
            # 2-D symmetric normalized gaussian function
            gaussian = 1/(2*np.pi*width[i]**2)*np.exp(-(x_**2 + y_**2)/(2*width[i]**2))
            gaussian_norm = np.trapz(np.trapz(gaussian**2, dx=dx),dx=dx)

            integrated_value = np.trapz(np.trapz(far_field * gaussian, dx = dx), dx=dx)
            overlap_integral_array[i] = integrated_value**2/(far_field_norm*gaussian_norm)
            
        width_max = width[np.argmax(overlap_integral_array)]
        overlap_with_NA = overlap_integral_array[np.argmin(np.abs(NA - width))]

        if(bool_plot):
            plt.figure()
            plt.plot(width, overlap_integral_array, linewidth = 2)
            plt.grid(True)
            
            
            plt.title(f"Overlap = {np.round(overlap_with_NA*100,2)}% for w = {NA}", fontsize = 15)
            # plt.axvline(x = width_max, color = 'k', linestyle = '--')
            plt.axvline(x = NA, color = 'k', linestyle = '-.')
            plt.axhline(y = overlap_with_NA, color = 'k', linestyle = '-.')
            plt.xlabel("Gaussian width", fontsize = 15)
            plt.ylabel("Overlap Integral", fontsize = 15)
            plt.ylim(0, 1)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.show()

            gaussian = 1/(2*np.pi*width_max**2)*np.exp(-(x_**2 + y_**2)/(2*width_max**2))
            
            fig, (ax1, ax2) = plt.subplots(1,2)
            im = ax1.imshow(gaussian, extent=[x.min(), x.max(), y.min(), y.max()])
            im = ax2.imshow(far_field, extent=[x.min(), x.max(), y.min(), y.max()])

            phis = np.linspace(0, np.pi * 2, 201)
            ax1.plot(width_max * np.cos(phis), width_max * np.sin(phis), lw=2, color="w", linestyle = "--")
            ax2.plot(width_max * np.cos(phis), width_max * np.sin(phis), lw=2, color="w", linestyle = "--")
            ax2.yaxis.set_ticklabels([])
            plt.show()

            # plt.figure()
            # plt.plot(x, far_field[:,101//2])
            # plt.plot(x, gaussian[:,101//2])

        return overlap_with_NA
    
    if(ff_type == 'directional'):
        
        width = np.linspace(0, 1, 100)
        eta_array = np.zeros(len(width))
        
        x = np.linspace(-1, 1, 101)
        y = np.linspace(-1, 1, 101)
        
        for i in range(len(width)):   
            eta_array[i] = calculate_emission_into_NA(far_field, width[i], x, y)
        
        eta = eta_array[np.argmin(np.abs(NA - width))]
        
        if(bool_plot == True):    
            plt.figure()
            plt.plot(width, eta_array, linewidth = 2)
            plt.grid()            
            plt.title(f"{np.round(eta*100,2)}% emission into NA = {NA}", fontsize = 15)
            plt.axvline(x = NA, color = 'k', linestyle = '-.')
            plt.axhline(y = eta, color = 'k', linestyle = '-.')
            plt.xlabel("NA", fontsize = 15)
            plt.ylabel("Field into NA", fontsize = 15)
            plt.ylim(0, 1.05)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.show()
        return eta
    
def get_real_part_field(field):
    "Return the real part of the field array"

    field = np.squeeze(np.nan_to_num(field))

    field_angle = np.squeeze(np.angle(field)).T/np.pi
    field_abs = np.squeeze(np.abs(field)).T
    field_real = field_abs*np.cos(np.pi*field_angle)

    return field_angle, field_real

def run_FDTD(weights_name, gaussian_width):
    # set desired wavelength and frequency of cavity
    wavelength_range = np.linspace(0.5, 0.55, 50)
    freq0 = td.C_0/np.mean(wavelength_range)
    freq_range = td.C_0/wavelength_range

    Nx, Ny, alattice, hole_radius, slab_thickness, refractive_index, x_shift = L3_params_520()

    # Materials - air and silicon
    air = td.Medium()
    slab_medium = td.Medium(permittivity=refractive_index**2)
    
    x_pos, y_pos = phc_holes_array(Nx, Ny, alattice, "L3")
    x_pos[0] = x_pos[0] + x_shift

    head, tail = os.path.split(weights_name)
    
    file_name = rf'.\{head}\param_history\{tail[:-4]}_param_history.npy'
    param_history = np.load(file_name, allow_pickle=True)
    param_history = param_history.reshape(param_history.shape[0]//(2*len(y_pos)), 2*len(y_pos))
    parameters = param_history[-1]
    
    dx = parameters[0: len(x_pos)]*alattice
    dy = parameters[len(x_pos):]*alattice  
    
    Nx_final, Ny_final = 32, 32
    x_pos, y_pos, dx, dy = convert_phc_size(Nx, Ny, Nx_final, Ny_final, alattice, dx, dy)
    x_pos[0] = x_pos[0] + x_shift
    
    # Make L3 cavity design
    slab, holes = get_phc(Nx_final, Ny_final, x_pos, y_pos, alattice, hole_radius, slab_thickness, slab_medium, dx ,dy)

    # Simulation domain size (micron)
    size_z = 2
    sim_size = [(Nx_final + 5) * alattice, (Ny_final + 5) * alattice * np.sqrt(3) / 2, size_z]

    # Initialize dipole source 
    source = return_dipole_source(freq0, fwidth = 2e14, center_location = [0, 0, 0], polarization = "Ey")
    
    # Initialize time series monitor for Q analysis
    time_series_monitor = get_time_series_monitor(t_start = 5e-13, center = [0, 0, 0], size = [0, 0, 0], name = "time_series")
    
    # Initialize far-field monitor
    far_field_monitor = get_far_field_monitor(slab_thickness, freq_range, t_start = 5e-13, width = 2e-13, name = "far_field monitor")
        
    # Initialize the grid for meshing
    grid_spec = define_mesh(alattice, steps_per_unit_length=17)
    
    # Simulation run time (s)
    run_time = 3e-12

    # Create simulation object
    L3_cavity_simulation = td.Simulation(
                            size=sim_size,
                            grid_spec=grid_spec,
                            structures=[slab, holes],
                            sources=[source],
                            monitors=[time_series_monitor, far_field_monitor],
                            run_time=run_time,
                            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
                            symmetry=(1, -1, 1),
                            shutoff = 10**-7
                            )
    
    # Run simulation
    sim_data = web.run(L3_cavity_simulation, task_name=weights_name, verbose=True)

    # save simulation object
    sim_data.to_file(fname = f'.\Tidy3D_data\{tail[:-4]}_phcsize_{32}.hdf5')
    
    # Analyze time monitor for quality factor
    index, quality_factor, resonant_wavelength = analyse_time_monitor_data(sim_data, freq_range, freq_window = (freq_range[-1], freq_range[0]), print_data = True)

    print("Resonant wavelength = ", resonant_wavelength, "nm, Quality factor = ", quality_factor)
    print("Closest wavelength = ", np.round(convert_freq_to_wavelength(freq_range[index])*1000,2), "nm")


    # Analyze far field monitor data

    far_field = np.squeeze(sim_data["far_field monitor"].power[:, :, :, index])
                  
    Ey_angle, Ey_real = get_real_part_field(sim_data["far_field monitor"].fields_cartesian.Ey[:, :, :, index].T)

    ux = np.array(far_field.ux)
    uy = np.array(far_field.uy)
    fig, ax = plt.subplots()
    
    im = ax.imshow(Ey_real/np.max(Ey_real), extent=[ux.min(), ux.max(), uy.min(), uy.max()])
    fig.colorbar(im, ax=ax, label = "$|E|^2$")

    x_gvec, y_gvec = return_reciprocal_vectors_520nm()

    for i in range(len(x_gvec)):
        plt.scatter(x_gvec[i], y_gvec[i], marker = 'x', color = "white", s=40)

    ax.set_xlabel("$k_x/k$")
    ax.set_ylabel("$k_y/k$")
    plt.title(f'Q = {np.round(quality_factor)}, $\lambda_0$ = {resonant_wavelength} nm')
    plt.savefig(f'./Tidy3D_data/{tail}_FDTD_result.png')
    plt.show()
    

    if(gaussian_width == 0):
        overlap_integral = calculate_directionality(far_field, ff_type = 'directional', NA = 0.55, bool_plot = True)
    else:
        overlap_integral = calculate_directionality(Ey_real, ff_type = 'gaussian', NA = 1.65, bool_plot = True)
        
        
def run_FDTD_780nm(weights_name, gaussian_width):
    # set desired wavelength and frequency of cavity
    wavelength_range = np.linspace(0.73, 0.8, 50)
    freq0 = td.C_0/np.mean(wavelength_range)
    freq_range = td.C_0/wavelength_range

    Nx, Ny, alattice, hole_radius, slab_thickness, refractive_index, x_shift = L3_params_780()

    # Materials - air and silicon
    air = td.Medium()
    slab_medium = td.Medium(permittivity=refractive_index**2)
    GaAs_refractive_index = td.Medium(permittivity=3.6**2)
    
    x_pos, y_pos = phc_holes_array(Nx, Ny, alattice, "L3")
    x_pos[0] = x_pos[0] + x_shift

    head, tail = os.path.split(weights_name)
    
    file_name = rf'.\{head}\param_history\{tail[:-4]}_param_history.npy'
    param_history = np.load(file_name, allow_pickle=True)
    param_history = param_history.reshape(param_history.shape[0]//(2*len(y_pos)), 2*len(y_pos))
    parameters = param_history[-1]
    
    dx = parameters[0: len(x_pos)]*alattice
    dy = parameters[len(x_pos):]*alattice  
    
    Nx_final, Ny_final = 32, 32
    x_pos, y_pos, dx, dy = convert_phc_size(Nx, Ny, Nx_final, Ny_final, alattice, dx, dy)
    x_pos[0] = x_pos[0] + x_shift
    
    # Make L3 cavity design
    GaAs_thickness = 5e-3
    # Initialize structures
    slab = td.Structure(geometry=td.Box(center=[0, 0, 0], size=[td.inf, td.inf, slab_thickness]), medium= slab_medium)
    GaAs_slab_up = td.Structure(geometry=td.Box(center=[0, 0, slab_thickness/2+GaAs_thickness/2], size=[td.inf, td.inf, GaAs_thickness]), medium= GaAs_refractive_index)
    GaAs_slab_down = td.Structure(geometry=td.Box(center=[0, 0, -slab_thickness/2-GaAs_thickness/2], size=[td.inf, td.inf, GaAs_thickness]), medium= GaAs_refractive_index)
    
    holes_group = []
    
    # Get the position of holes for L3 cavity
    nx, ny = Nx//2 + 1, Ny//2 + 1

    # Apply holes symmetrically in the four quadrants
    for ic, x in enumerate(x_pos):
        yc = y_pos[ic] if y_pos[ic] == 0 else y_pos[ic] + dy[ic]
        xc = x if x == 0 else x_pos[ic] + dx[ic]
        
        holes_group.append(td.Cylinder(center=[xc, yc, 0], radius=hole_radius, length=slab_thickness+GaAs_thickness*2))      
        if nx-alattice/2+0.1 > x_pos[ic] > 0 and (ny-alattice+0.1)*np.sqrt(3)/2 > y_pos[ic] > 0:
            holes_group.append(td.Cylinder(center=[-xc, -yc, 0], radius=hole_radius, length=slab_thickness+GaAs_thickness*2))      
        if nx-1.5*alattice+0.1 > x_pos[ic] > 0:
            holes_group.append(td.Cylinder(center=[-xc, yc, 0], radius=hole_radius, length=slab_thickness+GaAs_thickness*2))      
        if (ny-alattice+0.1)*np.sqrt(3)/2 > y_pos[ic] > 0 and nx-alattice+0.1 > x_pos[ic]:
            holes_group.append(td.Cylinder(center=[xc, -yc, 0], radius=hole_radius, length=slab_thickness+GaAs_thickness*2))     

    holes = td.Structure(geometry=td.GeometryGroup(geometries=holes_group), medium=td.Medium())

    # Simulation domain size (micron)
    size_z = 2
    sim_size = [(Nx_final + 5) * alattice, (Ny_final + 5) * alattice * np.sqrt(3) / 2, size_z]

    # Initialize dipole source 
    source = return_dipole_source(freq0, fwidth = 2e14, center_location = [0, 0, 0], polarization = "Ey")
    
    # Initialize time series monitor for Q analysis
    time_series_monitor = get_time_series_monitor(t_start = 5e-13, center = [0, 0, 0], size = [0, 0, 0], name = "time_series")
    
    # Initialize far-field monitor
    far_field_monitor = get_far_field_monitor(slab_thickness, freq_range, t_start = 5e-13, width = 2e-13, name = "far_field monitor")
        
    # Initialize the grid for meshing
    grid_spec = define_mesh(alattice, steps_per_unit_length=15)
    
    # Simulation run time (s)
    run_time = 3e-12

    # Create simulation object
    L3_cavity_simulation = td.Simulation(
                            size=sim_size,
                            grid_spec=grid_spec,
                            structures=[slab, GaAs_slab_up, GaAs_slab_down, holes],
                            sources=[source],
                            monitors=[time_series_monitor, far_field_monitor],
                            run_time=run_time,
                            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
                            symmetry=(1, -1, 1),
                            shutoff = 10**-7
                            )
    
    # Run simulation
    sim_data = web.run(L3_cavity_simulation, task_name=weights_name, verbose=False)

    # save simulation object
    sim_data.to_file(fname = f'.\Tidy3D_data\{tail[:-4]}_phcsize_{32}.hdf5')
    
    # Analyze time monitor for quality factor
    index, quality_factor, resonant_wavelength = analyse_time_monitor_data(sim_data, freq_range, freq_window = (freq_range[-1], freq_range[0]), print_data = True)

    print("Resonant wavelength = ", resonant_wavelength, "nm, Quality factor = ", quality_factor)
    print("Closest wavelength = ", np.round(convert_freq_to_wavelength(freq_range[index])*1000,2), "nm")


    # Analyze far field monitor data

    far_field = np.squeeze(sim_data["far_field monitor"].power[:, :, :, index])
                  
    Ey_angle, Ey_real = get_real_part_field(sim_data["far_field monitor"].fields_cartesian.Ey[:, :, :, index].T)

    ux = np.array(far_field.ux)
    uy = np.array(far_field.uy)
    fig, ax = plt.subplots()
    
    im = ax.imshow(Ey_real/np.max(Ey_real), extent=[ux.min(), ux.max(), uy.min(), uy.max()])
    fig.colorbar(im, ax=ax, label = "$|E|^2$")

    x_gvec, y_gvec = return_reciprocal_vectors_520nm()

    for i in range(len(x_gvec)):
        plt.scatter(x_gvec[i], y_gvec[i], marker = 'x', color = "white", s=40)

    ax.set_xlabel("$k_x/k$")
    ax.set_ylabel("$k_y/k$")
    plt.title(f'Q = {np.round(quality_factor)}, $\lambda_0$ = {resonant_wavelength} nm')
    plt.savefig(f'./Tidy3D_data/{tail}_FDTD_result.png')
    plt.show()
    

    if(gaussian_width == 0):
        overlap_integral = calculate_directionality(far_field, ff_type = 'directional', NA = 0.55, bool_plot = True)
    else:
        overlap_integral = calculate_directionality(Ey_real, ff_type = 'gaussian', NA = 1.65, bool_plot = True)