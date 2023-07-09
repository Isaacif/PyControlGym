import numpy as np
import matplotlib.pyplot as plt

class state_space:
    def __init__(self, method, *state_equation) -> None:
        self.state_equation = state_equation
        self.method = method
        self.order = len(self.state_equation)

        print(f"{self.order}º order system defined")

    def runSimulation(self, start, end, step, *initial_conditions):
        simulation_dots = (end-start)/step
        simulation_time = np.linspace(start, end, int(simulation_dots))
        state_space_solution = {}

        states_counter = 1
        for initial_value in initial_conditions:
            state_space_solution[f"x{states_counter}"] = np.array([initial_value])
            states_counter+=1
        
        if self.method == "euler":
            self.eulerSolver(step, simulation_time, state_space_solution)


    def eulerSolver(self, step, stime, solution):
        k = 0
        for time_step in stime:
            states_counter = 1
            state_vector = []
            for state in range(self.order):
                state_vector.append(solution[f"x{states_counter}"][k])
                states_counter+=1
            states_counter = 1
            for state in range(self.order):
                xin = solution[f"x{states_counter}"]
                xin = np.append(xin, xin[k]+(self.state_equation[states_counter-1](*state_vector))*step)
                solution[f"x{states_counter}"] = xin
                states_counter+=1
            k+=1
        
        self.solution = solution
        self.simulation_time = stime

    
    def plot2DResults(self, state):

        plt.plot(self.simulation_time, self.solution[f"x{state}"][:-1])
        plt.xlabel("Time in seconds")
        plt.ylabel(f"x{state} value")
        plt.show()
