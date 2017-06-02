import numpy as np
from adsorption_models import ExtendedLangmuir
import utility_functions_new as uf
from matplotlib import pyplot as plt
import matplotlib as mpl
from math import ceil
from tabulate import tabulate

vibs = [300, 750]
temp_range = [300]
m_pres = 700


def get_isotherms(max_p):
    sc_nta_rt = ExtendedLangmuir(name='NTA-Sc RT',
                                 e_ads=[-20.3, -15.8, -14.3],
                                 vibs=vibs,
                                 max_pressure=max_p,
                                 temp_limits=[298.15, 298.15])

    sc_nta_cold = ExtendedLangmuir(name='NTA-Sc Cold-RT',
                                   e_ads=[-20.3, -15.8, -14.3],
                                   vibs=vibs,
                                   max_pressure=max_p,
                                   temp_limits=[298.15, 233.15])  # -40 C

    sc_nta_hot = ExtendedLangmuir(name='NTA-Sc RT-HOT',
                                  e_ads=[-20.3, -15.8, -14.3],
                                  vibs=vibs,
                                  max_pressure=max_p,

                                  temp_limits=[353.15, 298.15])  # 80 C

    ca_nta_rt = ExtendedLangmuir(name='NTA-Ca RT',
                                 e_ads=[-11.5, -11.1, -11.1],
                                 vibs=vibs,
                                 max_pressure=max_p,
                                 temp_limits=[298.15, 298.15])

    ca_nta_cold = ExtendedLangmuir(name='NTA-Ca Cold-RT',
                                   e_ads=[-11.5, -11.1, -11.1],
                                   vibs=vibs,
                                   max_pressure=max_p,
                                   temp_limits=[298.15, 233.15])  # -40 C

    sc_nta_rt = sc_nta_rt.sequential_isotherm()
    sc_nta_hot = sc_nta_hot.sequential_isotherm()
    sc_nta_cold = sc_nta_cold.sequential_isotherm()
    ca_nta_rt = ca_nta_rt.parallel_iso()
    ca_nta_cold = ca_nta_cold.parallel_iso()

    return [sc_nta_rt, sc_nta_cold, sc_nta_hot, ca_nta_rt, ca_nta_cold]


def print_table(isotherms, p_min=5, p_max=100):

    table_data = list()

    for isoth in isotherms:
        tab_row = list()
        tab_row.append(isoth['name'])
        tab_row.append(uf.point_occupancy(isoth, p_min))
        tab_row.append(uf.point_occupancy(isoth, p_max))
        tab_row.append(uf.usable_occupancy(isoth,
                                           p_min=p_min,
                                           p_max=p_max))

        table_data.append(tab_row)

    table = tabulate(table_data, headers=['name',
                                          'p={}'.format(str(p_min)),
                                          'p={}'.format(str(p_max)),
                                          'UC'])

    return table


press = 100

occupancy_data = get_isotherms(max_p=press)

print(print_table(occupancy_data))

label = ['Sc 25 °C', 'Sc -40/25 °C', 'Sc 25/80 °C', 'Ca 25 °C', 'Ca -40/25 °C']

mpl.rcParams['font.size'] = 8
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1
mpl.rcParams['lines.linewidth'] = 0.8

# these variables are used to set the limits of
# the figures
x_max = 0.
y_max = 0.

line_style_list = ['solid', 'dashed', 'dotted', 'solid', 'dashed']
color_list = ['blue', 'blue', 'blue', 'red', 'red']

plt.subplot(211)

for i in range(len(occupancy_data)):
    plt.plot(occupancy_data[i]['pressure'],
             occupancy_data[i]['occ_sum'],
             label=label[i],
             linestyle=line_style_list[i],
             color=color_list[i])

    x_max = max(x_max, occupancy_data[i]['pressure'][-1])
    y_max = max(y_max, occupancy_data[i]['occ_sum'][-1])

plt.xlim([0, ceil(x_max * 2) / 2])
plt.ylim([0, ceil(y_max * 2) / 2])

# plt.xlabel('pressure [bar]', fontsize=8, color='black')
plt.ylabel('occupancy [unit-less]', fontsize=8, color='black')

plt.axvline(x=5.0, color='black', linestyle='dotted', linewidth=1.0)

# make the legend non-transparent
legend = plt.legend(framealpha=0.95)
legend.get_frame().set_linewidth(0.0)

plt.subplot(212)

# set new pressure
press = 700

occupancy_data_2 = get_isotherms(max_p=press)

print(print_table(occupancy_data_2, p_min=5, p_max=press))

for i in range(len(occupancy_data_2)):
    plt.plot(occupancy_data_2[i]['pressure'],
             occupancy_data_2[i]['occ_sum'],
             label=label[i],
             linestyle=line_style_list[i],
             color=color_list[i])

    x_max = max(x_max, occupancy_data_2[i]['pressure'][-1])
    y_max = max(y_max, occupancy_data_2[i]['occ_sum'][-1])

# the division by two trick
# is to be able to round to half-integers
plt.xlim([0, ceil(x_max * 2) / 2])
plt.ylim([0, ceil(y_max * 2) / 2])

plt.xlabel('pressure [bar]', fontsize=8, color='black')
plt.ylabel('occupancy [unit-less]', fontsize=8, color='black')

# make the legend non-transparent
legend = plt.legend(framealpha=0.95)
legend.get_frame().set_linewidth(0.0)

fig = plt.gcf()
fig.set_size_inches(3.5, 2.5 * 2)
fig.savefig('ads_isotherm_sc.png', dpi=600)

plt.show()
