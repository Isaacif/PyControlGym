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
        self.state_one = 1
        self.state_two = 2

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
                past_results = []
                for subs in range(self.order):
                    # Calculates x[k+1] = x[k] + dx_dt*time_step
                    past_results.append(solution[f"x{subs+1}"][-1])

                xind = equation(*past_results)
                xin = np.append(xin, xin[k]+(xind)*step)
                solution[f"x{states_counter}"] = xin
                states_counter+=1
            k+=1

        # Corrects array length        
        stime = np.append(stime, stime[-1])

        # Passes plot variables to class storage
        self.solution = solution
        self.simulation_time = stime

    def phasePortrait(self, x1, x2, state_one, state_two):        
        """
        Phase Portrait calculater, solves the values of two selected states for t=0

        Parameters
        ==========
        x1 : numpy array
            x1 plot interval.
        x2 : numpy array
            x2 plot interval.
        state_one : int
            first selected state.
        state_two : int
            second selected state.
        """

        # Creates meshgrid for the phase portrait
        self.X1, self.X2 = np.meshgrid(x1, x2)
        self.state_one = state_one 
        self.state_two = state_two 
        state_one = state_one - 1
        state_two = state_two - 1

        # Creates the u and v vectores, that indicate the rate of the derivates for both states
        self.u = np.zeros(self.X1.shape) 
        self.v = np.zeros(self.X2.shape)
        NI, NJ = self.X1.shape

        # result holds the grid sequence for the plot
        result = np.zeros(self.order)

        # main plot loop, for each pair i, j in the plot calculates the u, v vectors
        for i in range(NI):
            for j in range(NJ):
                x = self.X1[i, j]
                y = self.X2[i, j]
                xprime = []
                for equation in [self.state_equation[state_one], self.state_equation[state_two]]:
                    result[state_one] = x
                    result[state_two] = y
                    equation = sp.lambdify(self.symbols, equation.rhs, 'numpy')
                    xprime.append(equation(*result))

                # copies to the u, v holding variables
                self.u[i,j] = xprime[0]
                self.v[i,j] = xprime[1]

    def plot2DphasePortrait(self, xlim=[-2.25, 8.25], ylim = [-3.5, 4.5], solutions=False, 
                            tstart=0, tend=50, x1=[0], x2=[0]):
        """
        2D phase portrait plotter, can also show solutions for a the selected initial conditions

        Parameters
        ==========
        xlim : list
            x axe limites.
        ylim : list
            y axe limites.
        solutions : bool
            selects whether the path solutions will be plotted as well.
        tstart : int
            time start for the solutions plot.
        tend : int 
            time end for the solutions plot.
        x1 : list
            list of all desired initial values for x1 plot
        x2 : list
            list of all desired initial values for x2 plot
        """

        # Phase Portrait plot calls, quiver creates the vectors
        Q = plt.quiver(self.X1, self.X2, self.u, self.v, color='r')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.xlim(xlim)
        plt.ylim(ylim)

        # loop for the all the solutions, also creates the legend box to indicate all the start and end points
        if solutions:
            for x_x in x1:
                for x_y in x2:
                    plot_values = [x_x, x_y]
                    self.runSimulation(tstart, tend, 0.1, *plot_values, solve_method='int')
                    plt.plot(self.solution[f"x{self.state_one}"][:], self.solution[f"x{self.state_two}"][:], 'b-')
                    plt.plot(self.solution[f"x{self.state_one}"][0], self.solution[f"x{self.state_two}"][0], 'o', label=f"x1:{x_x} x2: {x_y} start")
                    plt.plot(self.solution[f"x{self.state_one}"][-1], self.solution[f"x{self.state_two}"][-1], 's', label=f"x1:{x_x} x2: {x_y} end")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1)
        
        plt.show()

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


