import numpy as np
import pybamm
from Prada2013_temp_dependent import get_parameter_values
import matplotlib.pyplot as plt



model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})

param = get_parameter_values()
param = pybamm.ParameterValues(param)  
#param = pybamm.ParameterValues("Prada2013")  

#ADOPTED from chen 2020...
param.update({
    "Negative current collector thickness [m]": 1.2e-05,
    "Positive current collector thickness [m]": 1.6e-05,
    "Negative current collector density [kg.m-3]": 8960.0,
    "Positive current collector density [kg.m-3]": 2700.0,
    "Negative current collector conductivity [S.m-1]": 58411000.0,
    "Positive current collector conductivity [S.m-1]": 36914000.0,
    "Cell volume [m3]": 2.42e-05,
    "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
    "Cell cooling surface area [m2]": 0.00531,
    "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
    "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
    "Negative electrode density [kg.m-3]": 1657.0,
    "Positive electrode density [kg.m-3]": 3262.0,
    "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
    "Separator density [kg.m-3]": 397.0,
    "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
}, check_already_exists=False)

experiment = pybamm.Experiment([
    "Discharge at 1C until 3.0 V",
    "Charge at 1C until 4.2 V",
])

sim = pybamm.Simulation(model, parameter_values=param,  experiment=experiment, C_rate=0.01)
sol = sim.solve([0, 3600])


sto = model.variables["Negative electrode stoichiometry"]
print(type(sto))

t = np.linspace(0, 3600, 1000)
V_t = sol["Terminal voltage [V]"](t)

import matplotlib.pyplot as plt
plt.plot(t, V_t)
plt.xlabel("Time (sec)")
plt.ylabel("Terminal Voltage (V)")
plt.savefig("test_result")

