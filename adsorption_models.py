import numpy as np
import sympy as syp
import platform
import utility_functions_new as uf


class ExtendedLangmuir:
    """A class for calculation of adsorption isotherms
    using extended Langmuir approach
    
    
    Known limitations:
    
    The current implementation assumes that the vibrations that 
    are formed in the adsorption process are identical for all 
    adsorbed molecules.
    
    Ehud Tsivion, 2017
    """

    def __init__(self,
                 vibs,
                 e_ads,
                 s_ads=70.0,
                 max_pressure=100.0,
                 name='default',
                 temp_limits=np.array([500, 298.15])):

        """Initialize object
        
        params
        ------
        
        vibs : float array
            
            The vibrations that are formed in the adsorption process.
            Units [cm-1]
            
        e_ads : float array
        
            The adsorption energies of each adsorption site.
            Units [kJ/mol]
            
        s_ads : float
        
            adsorption entropy. In current implementation assumed to be 
            fixed for all adsorption sites, temperatures and adsorption energies.
            Units of entropy [kJ mol-1 K-1]. 
                                    
        max_pressure : float
        
            The maximum pressure. The minimum is always zero.
            Units [bar]
            
        name : string
        
            The name of the studied system.
            
        temp_limits : float array
            
            The lower and upper limit of the temperature range.
            Units [K]

        """

        self.vibs_ = np.array(vibs, dtype=float)
        self.temp_limits_ = np.array(temp_limits, dtype=float)

        # re-order energies
        # this is important because molecules
        # are adsorbed first to the strongly
        # interacting sites
        self.energy_ = np.sort(np.array(e_ads, dtype=float))
        self.n_sites = len(self.energy_)  # the number of adsorption sites
        self.max_p = np.float(max_pressure)
        self.ads_entropy = s_ads
        self.name_ = name

        # verify input

        if len(temp_limits) != 2:
            raise Exception('The temperature range is not in correct form: [min, max]')

    def parallel_iso(self, pres_step=0.2, heating='linear', prepare_vars=False):
        """Adsorption isotherm of the parallel process
        
        This model assumed several adsorption sites, each capable of
        adsorption of single molecule (of any kind).
        
        All molecules are adsorbed in parallel,
        which is assumed to be correct for the case where the adsorption
        energy of each individual molecule is independent of the others
        
        Each site is assumed a Langmuir behavior, and the total occupancy 
        of the model is simply sum over the occupancy of individual sites.
        
        params
        ------
        
        pres_step : float
        
            The step size of the isotherm between between maximum and minimum 
            pressures.
            Units [bar]
            
        heating : string
            
            Set the fashion that temperature us controlled in the system:
            
            1. 'linear' - The temperature varies linearly between 
            maximum and minimum. Lowest pressures should have higher 
            temperature.
            
        prepare_vars : boolean
        
            a flag to signal to stop the run and return the variables 
            that were generated.
            
            The idea is to prepare the variables to other types of adsorption
            isotherms such as "sequential adsorption".
            
        return
        ------
        
            Unfortunately, we have two options here:
            
            if prepare_vars = False: return a dictionary with pressure, temperature, 
            occupancy of the sites and total occupancy
            
            if prepare_vars = False: return pressure, temperature and 
            equilibrium constants
            
        """

        # generate pressure range
        # note! the ratio of temperature:pressure is 1:1
        pres_range = np.arange(0, self.max_p + pres_step, pres_step)

        # generate temperature range
        temp_low_p = self.temp_limits_[0]
        temp_high_p = self.temp_limits_[1]

        if temp_low_p < temp_high_p:
            print('Warning, the temperature at low pressure should be higher \n'
                  'than the temperature at high pressure - in order to release \n'
                  'molecules which are bound to strongly adsorbing sites')

        if heating.lower() == 'linear':

            temp_range = np.linspace(temp_low_p,
                                     temp_high_p,
                                     num=len(pres_range))

        else:
            raise Exception('Heating type not identified')

        # prepare array to hold free energy at each temperature
        # its structure is free_g[sites, temperature]
        free_g = np.zeros((len(self.energy_), len(temp_range)), dtype=np.float)

        ####################
        # Gibbs free energy
        ####################

        # iterate over temperatures
        for idx_t in range(len(temp_range)):

            temp = temp_range[idx_t]

            vib_int_e_temp = uf.vib_energy(self.vibs_, temp)[0]
            rt_term = uf.R * temp
            st_term = self.ads_entropy * temp / 1000.0  # divide by 1000 to covert from K to kJ

            # iterate over all adsorption sites
            for idx_e in range(len(self.energy_)):
                e_ads_term = self.energy_[idx_e]

                uf.vib_energy(self.vibs_, temp)

                enthalpy_term = e_ads_term + vib_int_e_temp - rt_term

                free_g[idx_e, idx_t] = enthalpy_term + st_term

        ########################
        # Equilibrium constants
        ########################

        eq_constants = np.empty_like(free_g)

        for idx_t in range(len(temp_range)):

            temp = temp_range[idx_t]

            for idx_e in range(len(self.energy_)):
                eq_constants[idx_e, idx_t] = uf.eq_const(free_g[idx_e, idx_t], temp=temp)

        ########################
        # Construct the isotherm
        ########################

        # hold the occupancy of each individual site
        # its dimension is is site_occupancy[site, pressure]
        # note that n(temperatures) = n(pressures)

        site_occupancy = np.zeros_like(eq_constants)

        if prepare_vars:
            return pres_range, temp_range, eq_constants

        # iterate over pressures
        for idx_p in range(len(pres_range)):

            # extract pressure and temperature
            pres = pres_range[idx_p]

            # calculate the capacity of each site, at each
            # given pressure/temperature

            # iterate over sites
            for idx_site in range(len(self.energy_)):
                eq_const = eq_constants[idx_site, idx_p]

                site_occupancy[idx_site, idx_p] = uf.langmuir_occ(p=pres, k=eq_const)

        results = {'pressure': pres_range,
                   'temperature': temp_range,
                   'site_occupancy': site_occupancy,
                   'occ_sum': site_occupancy.sum(axis=0),
                   'energies': self.energy_,
                   'name': self.name_}

        return results

    def sequential_isotherm(self, pres_step=0.1, heating='linear'):

        # use "parallel_isotherm" method to generate the
        # generate the data required to begin calculation
        pressures, temps, eq_consts = self.parallel_iso(pres_step=pres_step,
                                                        heating=heating,
                                                        prepare_vars=True)

        n_sites = len(self.energy_)

        lang_equation = gen_ext_langmuir(n_sites)

        # this variable hold the occupancy of the model
        # that is, of all the sites together.
        occupancy = np.zeros_like(pressures)

        """
        Fast f2py evaluation of the Extended Langmuir expression
        couldn't make it work on Windows. Tested on Ubuntu, by would 
        probably work on also MacOS 
        """
        if platform.system() is not 'Windows':

            from sympy.utilities.autowrap import ufuncify

            """
            Because the free_symbols thing prints the sympy symbols 
            in a disordered way - we need a way to systematically
            order them
            
            It is assumed that the symbols can be either 'p' or 'Kx'
            where x is 1..n where n is an integer with the number of
            adsorbed molecules
            """

            symbols_list = np.zeros(self.n_sites + 1).tolist()

            for sym in lang_equation.free_symbols:

                if str(sym) == 'p':
                    position = 0
                else:
                    position = int(str(sym)[1:])

                symbols_list[position] = sym

            # this is set to use f95/f2py
            # haven't tried other combinations
            lang_func = ufuncify(symbols_list, lang_equation,
                                 language='f95',
                                 backend='f2py')

            """
            Prepare input argument list
            """
            arguments_list = np.zeros(self.n_sites + 1).tolist()

            arguments_list[0] = pressures

            for idx_eq_k in range(self.n_sites):
                arguments_list[idx_eq_k + 1] = eq_consts[idx_eq_k, :]

            occupancy = lang_func(*arguments_list)

        else:

            print('Comment: it seems that you are using sympy\'s Subs/evalf fuction\n'
                  'which is very slow. In non-windows you can use much faster method\n\n'
                  'BTW you are welcome to hack into this and make it work')

            for idx_p in range(len(pressures)):

                # we need to generate an array
                # to assign into the symbolic adsorption polynomial
                # assign the pressure
                assignment_array = [('p', pressures[idx_p])]

                # append expression for each equilibrium constant K
                for idx_eq_k in range(n_sites):
                    assignment_array.append(('K' + str(idx_eq_k + 1),
                                             eq_consts[idx_eq_k, idx_p]))

                occupancy[idx_p] = lang_equation.subs(assignment_array).evalf()

        results = {'pressure': pressures,
                   'temperature': temps,
                   'occ_sum': occupancy,
                   'energies': self.energy_,
                   'name': self.name_}

        return results


def gen_ext_langmuir(n_sites):
    """Generate the extended Langmuir model 
    for sequential adsorption to n sites
    
    params
    ------
    
    n_sites : integer
    
        The number of adsorption sites in the model
    
    return
    ------
        
        [a]
        
        ext_langmuir_eq : 
        
    """

    #####################################
    # Generation of adsorption polynomial
    #####################################

    ads_polynomial = 1

    eq_constant_string = ''

    for site in range(1, n_sites + 1):
        eq_constant_string += 'K{} '.format(site)

    k_array = syp.symbols(eq_constant_string)

    p = syp.symbols('p')

    for site in range(0, n_sites):

        K_expression = syp.symbols('K1')

        for i in range(1, site + 1):
            K_expression *= k_array[i]

        ads_polynomial += K_expression * p ** (site + 1)

    ###################################
    # Derive Extended Langmuir equation
    ###################################

    ext_langmuir_eq = p * syp.diff(ads_polynomial, p) / ads_polynomial

    return ext_langmuir_eq
