import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# Create gas object with GRI-Mech 3.0 mechanism
gas = ct.Solution('gri30.yaml')

# Set initial state: methane-air mixture at 1 atm, 1000 K
gas.TPX = 1800.0, ct.one_atm, 'CH4:1, O2:2, N2:7.52'

# Create a constant-volume reactor
reactor = ct.Reactor(gas)
sim = ct.ReactorNet([reactor])

# Time parameters
t_end = 0.00012  # seconds
dt = t_end / 100  # seconds
times = []
X_CH4 = []
X_O2 = []
X_CO2 = []
X_H2O = []

t = 0.0
while t < t_end:
    sim.advance(t)
    times.append(t)
    X_CH4.append(reactor.thermo['CH4'].X[0])
    X_O2.append(reactor.thermo['O2'].X[0])
    X_CO2.append(reactor.thermo['CO2'].X[0])
    X_H2O.append(reactor.thermo['H2O'].X[0])
    t += dt

# Print final mole fractions
print(f"Final CH4: {X_CH4[-1]:.4e}, O2: {X_O2[-1]:.4e}, CO2: {X_CO2[-1]:.4e}, H2O: {X_H2O[-1]:.4e}")

# Plot evolution
plt.plot(times, X_CH4, label='CH4')
plt.plot(times, X_O2, label='O2')
plt.plot(times, X_CO2, label='CO2')
plt.plot(times, X_H2O, label='H2O')
plt.xlabel('Time [s]')
plt.ylabel('Mole Fraction')
plt.legend()
plt.title('Species Evolution in Methane-Air Combustion')
plt.tight_layout()
plt.show()