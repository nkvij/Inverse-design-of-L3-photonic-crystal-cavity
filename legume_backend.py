import numpy as np
import matplotlib.pyplot as plt
import os
import autograd.numpy as npa

import legume

legume.set_backend('autograd')


def L3_params_936():
    """
    Parameters for L3 photonic crystal with resonance wavelength at 936 nm, 
    normalized by the lattice constant.

    Returns
    -------
    hole_radius: float
        radius of holes
    slab_thickness: float
        thickness of L3 photonic crystal slab
    refractive_index: float
        refractive index of slab
    x_shift: float
        shift in x position of the nearest hole to the cavity
    """

    hole_radius = 0.313
    slab_thickness = 0.826
    refractive_index = 3.55
    x_shift = 0.178

    return hole_radius, slab_thickness, refractive_index, x_shift


def L3_params_520():
    """
    Parameters for L3 photonic crystal with resonance wavelength at 520 nm, 
    normalized by the lattice constant.

    Returns
    -------
    hole_radius: float
        radius of holes
    slab_thickness: float
        thickness of L3 photonic crystal slab
    refractive_index: float
        refractive index of slab
    x_shift: float
        shift in x position of the nearest hole to the cavity
    """

    hole_radius = 0.3
    slab_thickness = 1.081
    refractive_index = 2
    x_shift = 0.26

    return hole_radius, slab_thickness, refractive_index, x_shift


def phc_cavity_holes_array(cavity_name, Nx, Ny):
    """
    Generate lists containing x and y coordinates for the hole centers for a 
    photonic crystal cavity

    Parameters
    ----------
    cavity_name : str
        PhC for which hole positions are to be calculated
    Nx : int
        # of holes in along the x-axis
    Ny : int
        # of holes along the y-axis

    Returns
    -------
    x_pos: list
        List containing the x coordinates of the center of the holes
    y_pos: list
        List containing the y coordinates of the center of the holes
    """
    x_pos, y_pos = [], []
    nx, ny = Nx // 2 + 1, Ny // 2 + 1

    if (cavity_name == 'L3'):
        for iy in range(ny):
            for ix in range(nx):
                if (ix == 0 and iy == 0
                        or ix == 1 and iy == 0):  # skip holes for L3 cavity
                    continue
                else:
                    x_pos.append(ix + (iy % 2) * 0.5)
                    y_pos.append(iy * np.sqrt(3) / 2)
    else:
        raise NotImplementedError(
            "PhC geometries apart from L3 not yet supported")

    return x_pos, y_pos


def design_phc(Nx, Ny, x_pos, y_pos, hole_radius, slab_thickness,
               refractive_index, dx, dy):
    """
    Using the PhotonicCryst class from Legume, create the design for L3 PhC by 
    initializing a Lattice with hexagonal lattice constants. Append holes to 
    the lattice by adding holes at positions (x_pos, y_pos), displaced by (dx, dy)

    Parameters
    ----------
    Nx : int
        # of holes along the x-axis
    Ny : int
        # of holes along the y-axis
    x_pos: list
        y coordinates for hole center
    y_pos: list
        y coordinates for hole center
    hole_radius: float
        radius for holes in the PhC
    slab_thickness: float
        thickness of PhC slab
    refractive_index: float
        refractive index of the PhC slab
    dx: numpy array
        x coordinate displacement for hole centers
    dy: numpy array
        y coordinate displacement for hole centers
    
    Returns
    -------
    L3_phc: PhotCryst object
    """

    # Initialize a lattice and PhC
    lattice = legume.Lattice([Nx, 0], [0, Ny * np.sqrt(3) / 2])
    L3_phc = legume.PhotCryst(lattice)

    # Add a layer to the PhC
    L3_phc.add_layer(d=slab_thickness, eps_b=refractive_index**2)

    # Get the position of holes for L3 cavity
    nx, ny = Nx // 2 + 1, Ny // 2 + 1

    # Apply holes symmetrically in the four quadrants. The numbers 0.6 and 1.1
    # ensure that holes with center at x=0 or y=0 are not initialized
    # more than necessary
    for ic, x in enumerate(x_pos):
        yc = y_pos[ic] if y_pos[ic] == 0 else y_pos[ic] + dy[ic]
        xc = x if x == 0 else x_pos[ic] + dx[ic]
        L3_phc.add_shape(legume.Circle(x_cent=xc, y_cent=yc, r=hole_radius))
        if nx - 0.6 > x_pos[ic] > 0 and (ny -
                                         1.1) * np.sqrt(3) / 2 > y_pos[ic] > 0:
            L3_phc.add_shape(
                legume.Circle(x_cent=-xc, y_cent=-yc, r=hole_radius))
        if nx - 1.6 > x_pos[ic] > 0:
            L3_phc.add_shape(
                legume.Circle(x_cent=-xc, y_cent=yc, r=hole_radius))
        if (ny -
                1.1) * np.sqrt(3) / 2 > y_pos[ic] > 0 and nx - 1.1 > x_pos[ic]:
            L3_phc.add_shape(
                legume.Circle(x_cent=xc, y_cent=-yc, r=hole_radius))

    return L3_phc


def gme_cavity(phc, Nx, Ny, gaussian_width, gmax, kpoints, gme_options):
    """
    Wrapper function for using GME class in Legume to simulate the PhC mode, calculate 
    its farfield and compute the gaussian overlap with target gaussian_width.
    The cavity mode is assumed to be at the band edge with index = Nx*Ny

    Parameters
    ----------
    phc : PhotCryst object
        PhC for which GME is computed
    Nx : int
        # of holes in along the x-axis
    Ny : int
        # of holes along the y-axis
    gaussian_width : float
        Target gaussian width for the farfield
        gaussian_width = 0 implies all losses are concentrated along kx = ky = 0 vectors
    gmax : float
        Cutoff for the g-vectors in GME computation
    kpoints : numpy array
        Reciprocal lattice vectors (kx, ky) for added resolution in the far field
    gme_options : dict
        Dictionary containing various options for GME computation

    Returns
    -------
    gme: GuidedModeExp class object
        Variable containing the result of GME computation
    quality_factor: float
        quality factor for L3 photonic crystal
    directionality: float
        Overlap of the farfield with a gaussian. Value between (0, 1)
    indmode: int
        Index of the cavity mode among the calculated eigenmodes
    """

    gmax = gmax + 0.01  # to avoid warnings

    # Initialize GME
    gme = legume.GuidedModeExp(phc, gmax=gmax)

    # Solve for the real part of the frequencies
    gme.run(kpoints=kpoints, **gme_options)

    # Calculate imaginary frequencies that correspond to cavity losses
    f_imag = []

    indmode = Nx * Ny

    overlap, f_imag = calculate_far_field(gme, kpoints, indmode,
                                          gaussian_width)
    # Calculate the quality factor:
    f_imag = npa.array(f_imag)
    quality_factor = gme.freqs[0, indmode] / 2 / npa.mean(f_imag)

    return (gme, quality_factor, overlap, indmode)


def gme_cavity_dot(phc, gaussian_width, gmax, kpoints, gme_options):
    """
    Wrapper function for using GME class in Legume to simulate the PhC mode, calculate 
    its farfield and compute the gaussian overlap with target gaussian_width.
    The cavity mode is identified by computing the dot product between the known cavity
    mode eigenvenctor (eigvec) and the eigenvectors returned by gme computation. 

    Parameters
    ----------
    phc : PhotCryst object
        PhC for which GME is computed
    gaussian_width : float
        Target gaussian width for the farfield
        gaussian_width = 0 implies all losses are concentrated along kx = ky = 0 vectors
    gmax : float
        Cutoff for the g-vectors in GME computation
    kpoints : numpy array
        Reciprocal lattice vectors (kx, ky) for added resolution in the far field
    gme_options : dict
        Dictionary containing various options for GME computation

    Returns
    -------
    gme: GuidedModeExp class object
        Variable containing the result of GME computation
    quality_factor: float
        quality factor for L3 photonic crystal
    directionality: float
        Overlap of the farfield with a gaussian. Value between (0, 1)
    indmode: int
        Index of the cavity mode among the calculated eigenmodes
    """

    # Load the eigenvector numpy array
    eigvec = np.load(f'./Legume_eigenvectors/L3_520nm_eigvecs_gmax={gmax}.npy',
                     allow_pickle=True)

    gmax = gmax + 0.01  # to avoid warnings

    gme = legume.GuidedModeExp(phc, gmax=gmax)  # Iniialize GME

    gme.run(kpoints=kpoints, **gme_options)  # run GME solver

    # Identify the index of the cavity mode
    dot_product = npa.abs(npa.dot(eigvec, npa.asarray(gme.eigvecs[0])))

    try:
        indmode = npa.nonzero(dot_product > 0.3)[0][0]

        overlap, f_imag = calculate_far_field(gme, kpoints, indmode,
                                              gaussian_width)
        # Calculate the quality factor:
        f_imag = npa.array(f_imag)
        quality_factor = gme.freqs[0, indmode] / 2 / npa.mean(f_imag)

        return (gme, quality_factor, overlap, indmode)
    except:
        raise ValueError(
            f"Mode not found! max(dot_product) = {np.max(dot_product)}")


def calculate_far_field(gme, kpoints, indmode, gaussian_width):
    """
    Function to calculate the farfield of the L3 PhC after GME computation is done.
    The farfield is then used to calculate overlap of the farfield with a gaussian of 
    zero mean and width = gaussian_width

    Parameters
    ----------
    gme: GuidedModeExp class object
        Variable containing the result of GME computation
    kpoints: numpy array of size (2, nkx*nky)
        (kx, ky) displacements of the reciprocal lattice vectors 
    indmode: int
        Index of the cavity mode among the calculated eigenmodes
    gaussian_width : float
        Target gaussian width for the farfield
        gaussian_width = 0 implies all losses are concentrated along kx = ky = 0 vectors
    
    Returns
    -------
    overlap: float
        overlap of the farfield with a gaussian of zero mean and width = gaussian_width
    f_imag_array: numpy array
        Array containing imaginary part of the cavity mode frequency; used to calculate
        the quality factor
    """

    f_imag_array = []  # loss component of the mode

    # lists to store farfield (rad_coupling_array) and reciprocal space vectors
    rad_coupling_array = []
    kx_array, ky_array = [], []

    if (gaussian_width != 0):
        # Gaussian far-field
        for i in range(kpoints[0, :].size):
            (freq_im, rad_coup, rad_gvec) = gme.compute_rad(i, [indmode])
            f_imag_array.append(freq_im)

            kx_array.extend(rad_gvec['l'][0][0] + kpoints[0][i])
            ky_array.extend(rad_gvec['l'][0][1] + kpoints[1][i])
            # Sum contributions from TE and TM modes
            rad_coupling_array.extend(
                npa.abs(rad_coup['l_te'][0])**2 +
                npa.abs(rad_coup['l_tm'][0])**2)

        kx_array = npa.array(kx_array)
        ky_array = npa.array(ky_array)
        rad_coupling_array = npa.array(rad_coupling_array)

        overlap = calculate_gaussian_overlap(kx_array, ky_array,
                                             rad_coupling_array,
                                             gaussian_width)

    elif (gaussian_width == 0):
        # Directional far-field
        for i in range(kpoints[0, :].size):
            (freq_im, rad_coup, rad_gvec) = gme.compute_rad(i, [indmode])
            f_imag_array.append(freq_im)

            kx_array.extend(rad_gvec['l'][0][0] + kpoints[0][i])
            ky_array.extend(rad_gvec['l'][0][1] + kpoints[1][i])
            # Sum contributions from TE and TM modes
            rad_coupling_array.extend(
                npa.abs(rad_coup['l_te'][0])**2 +
                npa.abs(rad_coup['l_tm'][0])**2)

        kx_array = npa.array(kx_array)
        ky_array = npa.array(ky_array)
        rad_coupling_array = npa.array(rad_coupling_array)

        overlap = calculate_directionality(kx_array, ky_array,
                                           rad_coupling_array)

    return overlap, f_imag_array


def get_kpoints(Nx, Ny, nkx, nky):
    """
    Sample nkx and nky points in {kx, ky} space in a uniform grid. Spacing between 
    two reciprocal vectors is 2*pi/Nx along kx axis and 2*pi/Ny aling ky axis. 
    These reciprocal lattice vectors are the displacements to the rad_gvec and are
    used to increasing the sampling of the farfield

    Parameters
    ----------
    Nx : int
        # of holes along the x-axis
    Ny : int
        # of holes along the y-axis
    nkx : int
        # of points to sample along kx axis
    nky : int
        # of points to sample along ky axis

    Returns
    -------
    kpoints: numpy array of size (2, nkx*nky)
        (kx, ky) displacements of the reciprocal lattice vectors 
    """

    kx = npa.linspace(0, (nkx - 1) / nkx * 2 * npa.pi / Nx, nkx)
    ky = npa.linspace(0, (nky - 1) / nky * 2 * npa.pi / Ny / npa.sqrt(3) * 2,
                      nky)

    kxg, kyg = npa.meshgrid(kx, ky)
    kxg = kxg.ravel()
    kyg = kyg.ravel()

    kpoints = npa.vstack((kxg, kyg))
    return kpoints


def calculate_directionality(kx_array, ky_array, rad_coup):
    """
    For gaussian_width = 0, the directionality is the ratio of the radiative 
    coupling (farfield) closest to the origin (kx=ky=0) to the total sum of rad_coup

    Parameters
    ----------
    kx_array : numpy array
        x coordinated of reciprocal lattice vectors for which farfield (rad_coup) is 
        computed
    ky_array : numpy array
        y coordinated of reciprocal lattice vectors for which farfield (rad_coup) is 
        computed
    rad_coup : numpy array
        farfield at coordinatea (kx_array, ky_array)

    Returns
    -------
    directionality: float
        ratio of the rad_coup closest to the origin to the total sum of rad_coup
    """

    # index of the lattice vector closest to origin
    index_00 = npa.argmin(npa.abs(kx_array) + npa.abs(ky_array))
    rad_00 = rad_coup[index_00]

    rad_tot = npa.sum(rad_coup)

    directionality = rad_00 / rad_tot

    return directionality


def calculate_gaussian_overlap(kx_array, ky_array, rad_coup_array,
                               gaussian_waist):
    """
    Compute the overlap integral of the farfield with a gaussain of zero mean and width
    = gaussian_waist.
    Before the overlap integral computation, the farfield is converted from the 
    reciprocal coordinates to cartesian coordiates.

    Parameters
    ----------
    kx_array : numpy array
        x coordinated of reciprocal lattice vectors for which farfield (rad_coup) is 
        computed
    ky_array : numpy array
        y coordinated of reciprocal lattice vectors for which farfield (rad_coup) is 
        computed
    rad_coup_array : numpy array
        farfield at coordinatea (kx_array, ky_array)
    gaussian_waist : float
        Target gaussian width for the farfield

    Returns
    -------
    overlap: float
        Overlap integral computed in the cartesian coordinates
    """

    # Adding 0.01 to avoid division by 0
    k_norm = npa.sqrt(npa.max(kx_array**2 + ky_array**2)) + 0.01
    kx_array = kx_array / k_norm
    ky_array = ky_array / k_norm

    # 4 mm is the focal length of the objective lens
    x = 4 * kx_array / npa.sqrt(1 - kx_array**2)
    y = 4 * ky_array / npa.sqrt(1 - ky_array**2)

    # define target gaussian array
    target_gaussian = 1 / (npa.pi * gaussian_waist**2) * npa.exp(
        -(x**2 + y**2) / (gaussian_waist**2))
    target_gaussian_norm = target_gaussian / npa.sqrt(
        npa.sum(target_gaussian**2))

    # Normalise the radiative couplings array
    rad_norm = npa.sqrt(rad_coup_array) / npa.sqrt(npa.sum(rad_coup_array))

    # Calculate gaussian overlap
    overlap = npa.dot(rad_norm, target_gaussian_norm)**2

    return overlap


def load_params_from_history(file_name, length):
    """
    Function to parse the saved data file and return the parameter history from the
    training data

    Parameters
    ----------
    file_name : str
        Location of file
    length : int
        # of optimized holes

    Returns
    -------
    dx: numpy array
        Array containing optimized x coordinate displacements of the hole centers
    dy: numpy array
        Array containing optimized y coordinate displacements of the hole centers
    """

    param_history = np.load(file_name, allow_pickle=True)
    param_history = param_history.reshape(
        param_history.shape[0] // (2 * length), 2 * length)
    parameters = param_history[-1]

    dx = parameters[0:length]
    dy = parameters[length:]

    return dx, dy


def calculate_Q(freqs, f_imag, indmode):
    """
    Calculate the quality factor of the PhC given the real (freqs) and imaginary (f_imag)
    part of the cavity mode frequency

    Parameters
    ----------
    freqs : numpy array
        Array containing real part eigenfrequencies computed by GME
    f_imag : numpy array
        Array containing imaginary part eigenfrequencies computed by GME
    indmode : int
        Index of the cavity mode among the calculated eigenmodes

    Returns
    -------
    quality_factor: float
        quality factor for L3 photonic crystal
    """

    quality_factor = freqs[0, indmode] / 2 / npa.mean(f_imag)

    return quality_factor


def save_eigvecs(gme, indmode, filename):
    """
    Save the eigenvectors corresponding to the cavity mode as a numpy array
    This is then used to track the cavtiy mode across training epochs

    Parameters
    ----------
    gme: GuidedModeExp class object
        Variable containing the result of GME computation
    indmode : int
        Index of the cavity mode among the calculated eigenmodes
    filename : str
        Location to save the eigenmode array
    """

    eigvec = gme.eigvecs[0][:, indmode]
    np.save(filename, eigvec, allow_pickle=True)


def get_farfield(gme, mind: int, cladding='u'):
    """
        Calculate the far field in k-space

        Parameters
        ----------
        mind: int
            Far field of the `mind` mode is computed.
        cladding: str, optional
            Cladding upper('u')/lower('l') for which far field is computed
        
        Returns
        -------
        rad_coups: np.ndarray
            Total coupling to the radiative modes in the choice of cladding
        rad_gvecs: np.ndarray
            Normalized reciprocal lattice vectors in the choice of cladding 
            corresponding to rad_coup
        """

    num_kpoints = gme.kpoints.shape[1]
    rad_gvecs = [[], []]
    rad_coups = []

    # Loop over kpoints to calculate the couplings to the radiative
    # modes and reciprocal lattice vectors in the desired cladding
    for kind in range(num_kpoints):
        (_, rad_coup, rad_gvec) = gme.compute_rad(kind, [mind])

        rad_coups.extend(
            np.abs(rad_coup[cladding + '_te'][0])**2 +
            np.abs(rad_coup[cladding + '_tm'][0])**2)

        rad_gvecs[0].extend(rad_gvec[cladding][0][0] + gme.kpoints[0, kind])
        rad_gvecs[1].extend(rad_gvec[cladding][0][1] + gme.kpoints[1, kind])

    # Normalize the reciprocal lattice vectors to be in range (-1,1)
    rad_gvec_norm = 2 * np.pi * gme.freqs[0, mind]
    rad_gvecs = rad_gvecs / rad_gvec_norm

    return (np.array(rad_coups), np.array(rad_gvecs))


def visualize_far_field(gme, mind: int, cladding='u'):
    """
    Plot the far field for the structure
    
    Parameters
    ----------
    gme: GuidedModeExp
    mind: int
        Far field of the `mind` mode is computed.
        
    cladding: str, optional
        Cladding upper('u')/lower('l') for which far field is computed
        
    Returns
    -------
    fig : matplotlib figure object
        Figure object for the plot.
    """

    if (cladding != 'u' and cladding != 'l'):
        raise ValueError(
            "cladding can be 'u' for upper or 'l' for lower cladding")

    # Calculate far field
    (rad_coups, rad_gvecs) = get_farfield(gme, mind=mind, cladding=cladding)

    fig, ax = plt.subplots(constrained_layout=True)

    ax.scatter(rad_gvecs[0], rad_gvecs[1], c=rad_coups, cmap='viridis', s=50)
    ax.set_xlabel('$k_x$/k')
    ax.set_ylabel('$k_y$/k')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    if cladding == 'u':
        ax.set_title("Farfield for upper-cladding")
    elif cladding == 'l':
        ax.set_title("Farfield for lower-cladding")

    return fig


def plotting_with_weights(weights_name):
    """
    Function to parse and plot the training data from the saved training data file

    Parameters
    ----------
    weights_name : str
        Training data file
    """

    weights = np.load(weights_name, allow_pickle=True)

    (size, ) = weights.shape

    param_history = weights[0]

    loss_function = weights[1]
    t_elapsed = weights[2]
    directionality = weights[3]
    quality_factor = weights[4]

    epochs = np.arange(len(loss_function)) + 1

    fig = plt.figure(figsize=(15, 8))
    spec = fig.add_gridspec(ncols=1, nrows=size - 1)  # subplot grid

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
    plt.ylim([10**int(np.log10(np.min(directionality)) - 1), 2])

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
    plt.ylim([10**int(np.log10(np.min(loss_function)) - 2), 5])
    plt.xlabel('Epochs')

    head, tail = os.path.split(weights_name)
    plt.suptitle(tail[:-4])

    fig_save_file = f'./{head}/plots/' + tail[:-4] + '.png'
    fig.savefig(fig_save_file)

    param_save_file = f'./{head}/param_history/' + tail[:-4] + '_param_history.npy'
    np.save(param_save_file, param_history)

    plt.show()
