#
# creator: Isaac Lima, Kayann
# Date: 12/08/23/
#
# A python framework for control scheme design and simulation purposes
#
# this file defines the state_space class which is meant to create
# the state space representation for the differencial equations in the model of study
# this class provides a general way to represent the equations symbolic
# solve them using the different available methods as well as plot the desired results


''' Importing dependencis '''
import numpy as np                                                             
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import odeint
from sympy.physics.mechanics import dynamicsymbols, init_vprinting
from sympy import symbols, Eq, Function
from sympy.solvers.ode.systems import dsolve_system
from IPython.display import display

class state_space:
    """
    Class constructor 
    
    Parameters
    ==========
    symbols : Symbol
        sympy symbols for the state variables.
    state_equation : list
        list of sympy state equations.
    """
    def __init__(self, symbols, *state_equation) -> None:
        self.state_equation = state_equation
        self.symbols = symbols
        self.order = len(self.state_equation)
        self.t = sp.Symbol('t')

        print(f"{self.order}º order system defined")

    def runSimulation(self, start, end, step, *initial_conditions, solve_method='euler'):
        """
        Simulation execution function

        Parameters
        ==========
        start : float
            start value for simulation.
        end : float
            end value for simulation.
        step : float 
            simulation step pass
        initial_conditions : list
            simulation initial conditions.
        solve_method : string
            selects solver method from the options: analytical solution, LSODA algorithm or Euler method.
        """

        # defines how many steps will be calculated
        simulation_dots = (end-start)/step  
        # creates the time array       
        simulation_time = np.linspace(start, end, int(simulation_dots))     
        # A dictionary that will hold the array solution for all states
        state_space_solution = {}

        # Creating the arrays with the initial conditions
        states_counter = 1
        for initial_value in initial_conditions:
            state_space_solution[f"x{states_counter}"] = np.array([initial_value])
            states_counter+=1
        
        # Calls the euler solving method
        if solve_method == "euler":
            self.eulerSolver(step, simulation_time, state_space_solution)

        # Calls the analytical solving method
        if solve_method == 'ana':
            self.symSolve('analytical', initial_conditions, simulation_time, state_space_solution)

        # Calls the LSODA algorithm solving method
        elif solve_method == "int":
            self.symSolve('integral', initial_conditions, simulation_time, state_space_solution)

    def symSolve(self, mode, initial_conditions, stime, solution):
        """
        Solver using sympy methods

        Parameters
        ==========
        mode : string
            defines solver algorithm.
        initial_conditions : list
            simulation initial conditions.
        stime : numpy array
            simulation time array.
        solution : dictionary
            Holds all state solutions.
        """

        # inits sympy symbolic display
        init_vprinting()
        self.mode = mode 

        # Solves equations for analytical method
        if self.mode == 'analytical':
            equations = []
            constants = []
            self.analSolutions = []
            for equation in self.state_equation:
                equations.append(equation)
            
            # Calls dsolve_system for analytical solution of system of DEs
            self.analSolution = dsolve_system(equations)
            
            # Applies Initial conditions for the constantes evaluation
            for constant in range(len(self.state_equation)):
                constants.append(self.analSolution[0][constant].rhs.subs(self.t, 0) - initial_conditions[constant])
                
            constants = sp.solve(constants)

            # Displays final solution
            for constant in range(len(self.state_equation)):
                self.analSolutions.append(self.analSolution[0][constant].subs(constants))
                display(self.analSolutions[constant])
            
            # Uses analytical solution for plot generation
            for state in range(self.order):
                self.solution = np.array([[]])
                for tt in stime:
                    self.solution = np.append(self.solution, sp.N(self.analSolutions[state].rhs.subs(self.t, tt)))
                    solution[f"x{state+1}"] = self.solution
            
            # Passes plot variables to class storage
            self.simulation_time = stime
            self.solution = solution

        # Solves equations for LSODA method
        elif self.mode == 'integral':
            equations = []
            for equation in self.state_equation:
                equations.append(sp.lambdify(self.symbols, equation.rhs, 'numpy'))
            
            # Creates an internal function for the system definition
            def system(state, t):
                local_states = []
                for equation in equations:
                    local_states.append(equation(*state))
                return local_states
            
            # Calls odeint from sympy to invoke the solver
            numerical_s = odeint(system, initial_conditions, stime)
            for state in range(self.order):
                solution[f"x{state+1}"] = numerical_s[:, state]

            # Passes plot variables to class storage
            self.simulation_time = stime
            self.solution = solution


    def eulerSolver(self, step, stime, solution):
        """
        Euler Solver, belongs to the class of non sympy methods

        Parameters
        ==========
        step : float
            defines solver step.
        stime : numpy array
            simulation time array.
        solution : dictionary
            Holds all state solutions.
        """

        # interaction variable
        k = 0
        equations = []
        for equation in self.state_equation:
                equations.append(sp.lambdify(self.symbols, equation.rhs, 'numpy'))

        # Solver loop
        for time_step in stime:
            states_counter = 1
            for equation in equations:
                xin = solution[f"x{states_counter}"]
                xind  = equation(*self.symbols)
                for symbol, subs in zip(self.symbols, range(self.order)):
                    # Calculates x[k+1] = x[k] + dx_dt*time_step
                    xind = xind.subs(symbol, solution[f"x{subs+1}"][-1])

                xin_derivative = sp.N(xind)
                xin = np.append(xin, xin[k]+(xin_derivative)*step)
                solution[f"x{states_counter}"] = xin
                states_counter+=1
            k+=1

        # Corrects array length        
        stime = np.append(stime, stime[-1])

        # Passes plot variables to class storage
        self.solution = solution
        self.simulation_time = stime

    
    def plot2DResults(self, state):
        """
        2D time domain plotter abstraction.
        Creates a simple plot for the selected state.

        More elaborated plots can be done by acessing the solutions externally

        Parameters
        ==========
        state : int
           selects state to be ploted.
        """

        # Plots the selected state
        plt.plot(self.simulation_time, self.solution[f"x{state}"])
        plt.xlabel("Time in seconds")
        plt.ylabel(f"x{state} value")
        plt.show()


