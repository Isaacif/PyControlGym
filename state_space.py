import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols, init_vprinting
from sympy import symbols, Eq, Function
from sympy.solvers.ode.systems import dsolve_system
from IPython.display import display

class state_space:
    def __init__(self, method, *state_equation, symbolic=False) -> None:
        self.state_equation = state_equation
        self.method = method
        self.order = len(self.state_equation)
        self.symbolic = symbolic
        if self.symbolic:
            self.t = sp.Symbol('t')


        print(f"{self.order}º order system defined")
        print(f"symbolic: {self.symbolic}")

    def runSimulation(self, start, end, step, *initial_conditions, solve_method='num'):
        simulation_dots = (end-start)/step
        simulation_time = np.linspace(start, end, int(simulation_dots))

        if not(self.symbolic):
            state_space_solution = {}

            states_counter = 1
            for initial_value in initial_conditions:
                state_space_solution[f"x{states_counter}"] = np.array([initial_value])
                states_counter+=1
            
            if self.method == "euler":
                self.eulerSolver(step, simulation_time, state_space_solution)


        else:
            if solve_method == 'ana':
                self.symSolve('analytical', initial_conditions)
                self.simulation_time = simulation_time
            

    def symSolve(self, mode, initial_conditions):
        init_vprinting()
        self.mode = mode 
        if self.mode == 'analytical':
            equations = []
            constants = []
            self.analSolutions = []
            for equation in self.state_equation:
                equations.append(equation)
            
            self.analSolution = dsolve_system(equations)

            for constant in range(len(self.state_equation)):
                constants.append(self.analSolution[0][constant].rhs.subs(self.t, 0) - initial_conditions[constant])
                
            constants = sp.solve(constants)
            for constant in range(len(self.state_equation)):
                self.analSolutions.append(self.analSolution[0][constant].subs(constants))
                display(self.analSolutions[constant])



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

    
    def plot2DResults(self, mode, state):
        if mode == 'ana':
            self.solution = np.array([[]])
            for tt in self.simulation_time:
                self.solution = np.append(self.solution, sp.N(self.analSolutions[state].rhs.subs(self.t, tt)))
            plt.plot(self.simulation_time, self.solution)
            
        else:   
            plt.plot(self.simulation_time, self.solution[f"x{state}"][:-1])
            plt.xlabel("Time in seconds")
            plt.ylabel(f"x{state} value")
            plt.show()


