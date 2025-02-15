import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

# def lorenz(t, state, sigma=10, beta=8/3, rho=28):
#     x, y, z = state
#     dxdt = sigma * (y - x)
#     dydt = x * (rho - z) - y
#     dzdt = x * y - beta * z
#     return [dxdt, dydt, dzdt]

# # Initial conditions
# state0 = [0, 0, 0]
# time_span = (0, 10000)
# time_eval = np.linspace(time_span[0], time_span[1], 10000)

# for i in range(-50001, 50001):
    
#     #Scrambles x y and z starting conditions to generate a new system
#     state0[i % 3] == i
    
#     # Solve the system
#     sol = solve_ivp(lorenz, time_span, state0, t_eval=time_eval)

#     #Save the time series data of the x y and z axes as individual csvs
#     dfx = pd.DataFrame({'Time': sol.t, 'X': sol.y[0]})
#     dfx.to_csv("lorenz_output_x" + i.toString() + ".csv", index=False)
#     print("Data saved to lorenz_output_x" + i.toString() + ".csv")

#     dfy = pd.DataFrame({'Time': sol.t, 'Y': sol.y[1]})
#     dfy.to_csv("lorenz_output_y " + i.toString() + ".csv", index=False)
#     print("Data saved to lorenz_output_y " + i.toString() + ".csv")

#     dfz = pd.DataFrame({'Time': sol.t, 'Z': sol.y[2]})
#     dfz.to_csv("lorenz_output_z" + i.toString() + ".csv", index=False)
#     print("Data saved to lorenz_output_z" + i.toString() + ".csv")



# Plot the attractor
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(sol.y[0], sol.y[1], sol.y[2], color='b', lw=0.5)
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Lorenz Attractor")
# plt.show()


def logistic_equation(r, x):
    return r * x * (1 - x)

def simulate_logistic_equation(r, x0, num_steps):
  
    x_values = [x0]
    for _ in range(num_steps):
        x_values.append(logistic_equation(r, x_values[-1]))
    return x_values

r = 2.8  
x0 = 0.2 
num_steps = 100

x_values = simulate_logistic_equation(r, x0, num_steps)

plt.plot(range(num_steps + 1), x_values, marker='o', linestyle='-')
plt.xlabel("Time step")
plt.ylabel("Population (x)")
plt.title(f"Logistic Equation Simulation (r={r}, x0={x0})")
plt.grid(True)
plt.show()