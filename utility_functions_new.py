"""
Important constants
"""
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

R = np.float(0.008310488)  # gas constant in [kJ mol-1 K-1]
ch4_molar_mass = np.float(16.04)  # in [gr mol-1]
avogadro = np.float(6.02214086E23)  # in [mol -1]
ch4_density_stp = 0.716  # in [gr L-1]
plank_const = np.float(6.62607004E-37)  # Planck constant in [kJ sec]
boltz_const = np.float(1.38064852E-26)  # boltzmann constant in [kJ K-1]
speed_ol = np.float(29980000000)  # speed of light in [cm sec-1]


def vib_energy(frequency, temp):
    """The total vibrational energy as a function of frequency

    calculates the total vibrational energy at each given temperature

    params
    ------

    frequency : float array
        the frequency of the vibration in cm-1

    temp : float array
        the temperature in Kelvins

    returns
    -------

    energy : float array
        an array with internal energies of vibrations, at various temperature. 
        The shape of the array 'temp'

    """

    # vibrational temperature

    vtemp = plank_const * speed_ol / boltz_const * frequency

    vtemp = np.matrix(vtemp)

    temp = 1 / np.matrix(temp)

    # need to convert to array, because the element wise exponential
    # cannot work with matrix
    intermediate = np.array(vtemp.T * np.transpose(temp).T)

    # energy is a np.matrix, need to convert to array
    energy = R * vtemp * np.matrix((0.5 + (np.exp(intermediate) - 1) ** -1))
    energy = np.array(energy)[0]

    return energy


def gibbs(h, s, temp):
    """The Gibbs free energy

    params
    ------
    
    h : float 
        
        The enthalpy.
        Units [kJ/mol]
        
    s : float
    
        The entropy, given as S/R (experimental nomenclature)
        the multiplication by R is needed.
        
    temp : float
    
        the temperature.
        Units [K]
        
    """

    return h - s * temp * R


def calc_free_g(energies, temperatures):
    """The Gibbs free energy of adsorption
    
    
    params
    ------
    
        energies : float array
            
            the energies of adsorption of each adsorbed molecule.
            Units [kJ/mol]
            
        temperatures : float array
        
            The temperatures ...
    """
    pass


def eq_const(free_g, temp):
    """Adsorption reaction constants

    params
    ------

        free_g : float

            The Gibbs free energy

        temp : float

            The temperature
    """

    return np.exp(-free_g / (R * temp))


def langmuir_occ(p, k):
    """The Langmuir adsorption model
    
    Calculate the fraction of site that are occupied by adsorabte molecules.
    Alternatively - the average occupancy of a single site.
    
    reference
    ---------
    https://en.wikipedia.org/wiki/Langmuir_adsorption_model
    
    
    params
    ------
    
    p : float
        
        The pressure
        
    k : float
    
        The equilibrium constant for adsorption

    return
    ------
    
    occupancy : float
    
        The site occupancy, according to Langmuir theory.
        Unitless.
        
    """

    intermediate = k * p

    occupancy = intermediate / (intermediate + 1)

    return occupancy

def point_occupancy(isotherm, pressure):
    """extract the occupancy of the open-metal site at
    a given pressure

    params
    ------

    isotherm : dictionary
        the isotherm - a dictionary with isotherm data
        'pressure' - the pressure at each given point
        'occ_sum' - the occupancy at each given point

    pressure : float
        the pressure in which to evaluate occupancy
    """

    indx = np.where(isotherm['pressure'] >= pressure)[0][0]

    return isotherm['occ_sum'][indx]


def usable_occupancy(isotherm, p_min=5.0, p_max=100.0):
    """The usable occupancy of the adsorption system
    
    Calculate the usable occupancy of a single site
    this is defined as:
    
        U_oc = capacity(p_max) - capacity(p_min)
        
    To get the system usable capacity, you need to 
    multiply this number of the concentration of these 
    adsorption sites in the macroscopic system
    
    params
    ------
    
    return
    ------
    
    usable_occ : float
    
        The usable occupancy of the system. 
    
    """

    occ_max = point_occupancy(isotherm, p_max)
    occ_min = point_occupancy(isotherm, p_min)

    usable_occ = occ_max - occ_min

    return usable_occ


def plot_occupancy(occupancy_data, title=None, vline_min=5.0, vline_max=100.0, save=False, args=()):
    """Utility function to plot adsorption isotherms
    
    params
    ------
    
    occupancy data : float array
        an array which holds the data of occupancy. 
        If it holds multiple isotherm data, they will all be plotted.
    
    title : string
        The title
        
    vline_min : float
        plot a vertical line at the minimum pressure
        
    vline_max : float
        plot a vertical line at the maximum pressure
        
    dave : Boolean
        Flag to save as image

    """

    # http://matplotlib.org/users/customizing.html
    # mpl.rcParams['font.size'] = 12
    # mpl.rcParams['savefig.bbox'] = 'tight'
    # mpl.rcParams['savefig.pad_inches'] = 0.1
    # mpl.rcParams['lines.linewidth'] = 0.8

    # these variables are used to set the limits of
    # the figures
    x_max = 0.
    y_max = 0.

    plt.plot(occupancy_data['pressure'], occupancy_data['occ_sum'])

    x_max = max(x_max, occupancy_data['pressure'][-1])
    y_max = max(y_max, occupancy_data['occ_sum'][-1])

    # the division by two trick
    # is to be able to round to half-integers
    plt.xlim([0, ceil(x_max * 2) / 2])
    plt.ylim([0, ceil(y_max * 2) / 2])

    plt.xlabel('pressure [bar]')
    plt.ylabel('occupancy [unitless]')

    plt.axvline(x=vline_min, color='black', linestyle='dotted', linewidth=1.0)

    if vline_max < occupancy_data['pressure'][-1]:
        plt.axvline(x=vline_max, color='black', linestyle='dotted', linewidth=1.0)

    if title:
        plt.title(title)

    if save:
        fig = plt.gcf()
        fig.set_size_inches(3.5, 2.5)
        fig.savefig('ads_isotherm.png', dpi=600)

    plt.show()
