import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

class estimator:
    def __init__(self, file, delimit, *columns) -> None:
        self.data = pd.read_csv(file, delimiter=delimit)
        self.frameCol = {}

        for column in columns:
            self.frameCol[f"Var {column}"] = self.data[column].to_numpy() 


    def splitByCharacter(self, column, character, index):
        numerical_value = np.array([])
        for number in self.frameCol[f"Var {column}"]:
            numerical_value = np.append(numerical_value, float(number.split(character)[index]))
        
        self.frameCol[f"Var {column}"] = numerical_value

    
    def scaleByFactor(self, column, Factor):
        self.frameCol[f"Var {column}"] = Factor*self.frameCol[f"Var {column}"] 
        
    def makeLinear(self, column, minValue, maxValue, minSpace, makeItStandard):
        newlinear = np.array([])
        revolves = 0
        minSpace = 0
        i = 0
        for reading in self.frameCol[f"Var {column}"]:
            if reading < minValue and minSpace > 30:
                revolves+=1
                minSpace = 0
            else:
                minSpace+=1
            newValue = reading + revolves*maxValue
            newlinear = np.append(newlinear, newValue)

        if makeItStandard:
            self.frameCol[f"Var {column}"] = newlinear
        else:
            return newlinear
    
    def getVector(self, column):
        return self.frameCol[f"Var {column}"]


    def runRLS(self, alpha, order, inputs, output, limit):
        shape_x = (1, 1, order, 1)
        shape_w = (1, order, 1)
        P_covariance_Matrix = np.array([np.identity(order)*1/alpha])
        P_covariance_Matrix = np.append(P_covariance_Matrix, [np.identity(4)*1/alpha], axis=0)
        x_inputs_Matrix = np.zeros(shape_x, dtype=float)
        w_weights_Matrix = np.zeros(shape_w, dtype=float)

        n=1
        for interaction in output:
            x_n = np.empty(shape_x)
            x_n = inputs[n].reshape(x_n.shape)
            x_inputs_Matrix = np.append(x_inputs_Matrix, x_n, axis=0)
            T1 = np.matmul(P_covariance_Matrix[n], x_inputs_Matrix[n][0])
            T2 = np.matmul(np.transpose(x_inputs_Matrix[n][0]), P_covariance_Matrix[n])  
            T3 = np.matmul(np.transpose(x_inputs_Matrix[n][0]), T1)
            P_n = P_covariance_Matrix[n] - np.matmul(T1, T2)/(1+T3)
            P_covariance_Matrix = np.append(P_covariance_Matrix, [P_n], axis=0)
            g_n = np.matmul(P_n, x_inputs_Matrix[n][0])
            estimate_v = np.matmul(np.transpose(x_inputs_Matrix[n][0]), w_weights_Matrix[n-1])
            error = output[n+1] - estimate_v
            w_n = w_weights_Matrix[n-1] + g_n*error
            w_weights_Matrix = np.append(w_weights_Matrix, [w_n], axis=0)
            n+=1

            if n > limit:
                break
        
        self.weights = w_n
        return w_n
    
    def Plot_answer(self, time, order, inputs, output, initial_step, limit, plotStyle):
        shape_x = (1, 1, order, 1)
        estimated_t = np.array([output[0], output[1]])
        k = initial_step
        error = 0
        for interaction in output:
            x_n = np.empty(shape_x)
            x_n = inputs[k-2].reshape(x_n.shape)
            est = np.matmul(np.transpose(x_n[0][0]), self.weights)
            estimated_t = np.append(estimated_t, est)
            error+=est-output[k] 
            k+=1
            if k > limit:
                break
        print(error/k)

        plt.figure(1)
        if plotStyle == "plot":
            plt.plot(time[:limit], estimated_t[:limit])
            plt.plot(time[:limit], output[:limit])

        elif plotStyle == "scatter":
            plt.scatter(time[:limit], estimated_t[:limit])
            plt.scatter(time[:limit], output[:limit])
        
        plt.figure(2)

        if plotStyle == "plot":
            plt.plot(time[:limit], output[:limit]-estimated_t[:limit])

        elif plotStyle == "scatter":
            plt.scatter(time[:limit/4], output[:limit/4]-estimated_t[:limit/4])

        return estimated_t