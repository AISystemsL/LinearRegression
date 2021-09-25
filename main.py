# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sympy as sym
import numpy
import math
import sympy.plotting
from sympy import symbols
from sympy.plotting import plot
import numpy as np
import matplotlib.pyplot as plt

knownValues = [[1, 2.1], [2, 3.9], [3, 6.1], [4, 8.4], [5, 9.8]]
inIdx = 0
outIdx = 1
lambda_regularization = 0.0000000152 # ln(λ)=-18 # 0.005

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def f(x1, x2):
    return x1**2 + x1*(x2**2)


def GetErrorReg(w0, w1):
    error = 0
    for i in knownValues:
        error += (i[outIdx] - ((w1 * i[inIdx]) + w0))**2

    return error


def visual_E_partial_der(w0, w1, wrt):
    return sym.diff(GetErrorReg(w0, w1), wrt)


def getPartialDerivativeOfEwrtW0():
    E_prime_Lambdified = sym.lambdify([w0, w1], visual_E_partial_der(w0, w1, w0), "numpy")
    return E_prime_Lambdified


def getPartialDerivativeOfEwrtW1():
    E_prime_Lambdified = sym.lambdify([w0, w1], visual_E_partial_der(w0, w1, w1), "numpy")
    return E_prime_Lambdified


# def eval_f_prime_wrt_x1(x1, x2):
#    return visual_f_prime_wrt_x1(x1, x2).evalf(subs={x: 1, y: 1})


def eval_E_partial_der(w0, w1, wrt):
    return visual_E_partial_der(w0, w1, wrt).evalf(subs={w0: 1, w1: 1})


def printError(w0, w1):
    print(getError_AsString(w0, w1))


def getError_AsString(w0, w1):
    return 'E(w) = ' + str(GetErrorReg(w0, w1)) + ' w0=' + str(w0) + ' w1=' + str(w1)


def getLoss_AsString(w0, w1):
    return 'Loss(w) = ' + str(getLoss(w0, w1))


def getLoss(w0, w1):
    return GetErrorReg(w0, w1) + getRegularizationTerm(w0, w1)


def getRegularizationTerm(w0, w1):
    return 2 * lambda_regularization * (w0**2 * w1**2)


def print_E_gradient(w0, w1):
    print(get_E_gradient_AsString(w0, w1))


def get_E_gradient_AsString(w0, w1):
    gradient = get_E_gradient(w0, w1)
    return 'gradient: [' + str(gradient[0]) + ', ' + str(gradient[1]) + ']'


def get_E_gradient(w0, w1):
    derivativeOfEwrtW0 = getPartialDerivativeOfEwrtW0()
    derivativeOfEwrtW1 = getPartialDerivativeOfEwrtW1()
    return [derivativeOfEwrtW0(w0, w1), derivativeOfEwrtW1(w0, w1)]


def print_error_params_gradient(epoch, w0, w1):
    print('epoch ' + str(epoch) + ': ' + getError_AsString(w0, w1) + ' '
          + getLoss_AsString(w0, w1) + ' ' + get_E_gradient_AsString(w0, w1))


def getBestW1():
    sum_x_times_p = 0
    for i in knownValues:
        sum_x_times_p += i[inIdx] * i[outIdx]

    sum_x =getSum_x()
    sum_y = getSum_y()

    sum_x_power = 0;
    for i in knownValues:
        sum_x_power += i[inIdx] ** 2

    numerator = sum_x_times_p - ((1 / len(knownValues)) * sum_x * sum_y)
    denominator = sum_x_power - ((1 / len(knownValues)) * sum_x_power)

    bestW1 = numerator / denominator
    return bestW1

def getBestW0():
    bestW0 = ((1 / len(knownValues)) * getSum_y()) - (getBestW1() * ((1 / len(knownValues)) * getSum_x()))
    return bestW0


def getSum_x():
    sum_x = 0;
    for i in knownValues:
        sum_x += i[inIdx]
    return sum_x


def getSum_y():
    sum_y = 0;
    for i in knownValues:
        sum_y += i[outIdx]
    return sum_y


def gradientDescentAlg():
    currentW0 = 3.1
    currentW1 = 1.17
    eta = 0.01650

    epochs = 800
    converged = False
    MIN_ERROR = 0.21100000000026
    print('GRADIENT DESCENT')
    print('eta = ' + str(eta))

    error = GetErrorReg(currentW0, currentW1)
    firstEpochIdx = 0;
    error_vs_epochs = [error]
    regularization_vs_epochs = [50 * getRegularizationTerm(currentW0, currentW1)]

    print_error_params_gradient(firstEpochIdx, currentW0, currentW1)
    for current_epoch in range(epochs):
        if error < MIN_ERROR:
            converged = True
            print('converged in ' + str(current_epoch))
            break

        gradient = get_E_gradient(currentW0, currentW1)
        descending_gradient = [(-1 * gradient[0]), (-1 * gradient[1])]

        regularization_term = - 2 * lambda_regularization

        currentW0 += (eta * descending_gradient[0]) + (regularization_term * currentW0)
        currentW1 += (eta * descending_gradient[1]) + (regularization_term * currentW1)

        error = GetErrorReg(currentW0, currentW1)
        print_error_params_gradient(current_epoch, currentW0, currentW1)
        error_vs_epochs.append(error)
        regularization_vs_epochs.append(50 * getRegularizationTerm(currentW0, currentW1))

    return error_vs_epochs, regularization_vs_epochs


def showPlot():
    x = symbols('x')
    p1 = plot(x * x, show=False)
    p2 = plot(x, show=False)
    p1.append(p2[0])
    # p1
    # p1.show()

    p3 = sympy.plotting.plot3d(GetErrorReg(w0, w1), (w0, 0.08999, 0.09001), (w1, 9899999928, 9899999935))
    p3.show

def manualTestParams():

    print_error_params_gradient(3.1, 1.17)
    print_error_params_gradient(3.0, 1.169)
    print_error_params_gradient(2.8, 1.1695)
    print_error_params_gradient(2.6, 1.1699)
    print_error_params_gradient(2.59, 1.17)
    print_error_params_gradient(2.4, 1.4)
    print_error_params_gradient(2.3, 1.3)
    print_error_params_gradient(2.2, 1.4)
    print_error_params_gradient(2.1, 1.5)
    print_error_params_gradient(2.0, 1.4)
    print_error_params_gradient(1.94, 1.46)
    print_error_params_gradient(1.92, 1.47)
    print_error_params_gradient(1.90, 1.49)
    print_error_params_gradient(1.85, 1.495)
    print_error_params_gradient(1.80, 1.497)
    print_error_params_gradient(1.75, 1.499)
    print_error_params_gradient(1.73, 1.501)
    print_error_params_gradient(1.71, 1.505)
    print_error_params_gradient(1.67, 1.509)
    print_error_params_gradient(1.65, 1.515)
    print_error_params_gradient(1.60, 1.525)
    print_error_params_gradient(1.58, 1.535)
    print_error_params_gradient(1.55, 1.545)
    print_error_params_gradient(1.43, 1.585)
    print_error_params_gradient(1.33, 1.615)
    print_error_params_gradient(1.23, 1.635)
    print_error_params_gradient(1.13, 1.655)
    print_error_params_gradient(1.03, 1.675)
    print_error_params_gradient(1.04, 1.695)
    print_error_params_gradient(1.031, 1.715)
    print_error_params_gradient(1.030, 1.735)
    print_error_params_gradient(1.028, 1.725)
    print_error_params_gradient(1.029, 1.730)
    print_error_params_gradient(1.0285, 1.732)
    print_error_params_gradient(0.55, 1.92)
    print_error_params_gradient(0.50, 1.90)
    print_error_params_gradient(0.47, 1.89)
    print_error_params_gradient(0.46, 1.88)
    print_error_params_gradient(0.1, 1.987)
    print_error_params_gradient(0.1, 1.987)
    print(getBestW0())
    print(getBestW1())
    print_error_params_gradient(getBestW0(), getBestW1())
    print_error_params_gradient(0, 2)

if __name__ == '__main__':

    # x, y = sym.symbols('x y')
    # print(visual_f_prime_wrt_x1(x, y))  # This works.
    print('# of knownInputs: ' + str(len(knownValues)))
    printError(0, 0)

    w0, w1 = sym.symbols('w0 w1')
    print('E_prime_wrt_w0: ' + str(visual_E_partial_der(w0, w1, w0)))
    print('E_prime_wrt_w1: ' + str(visual_E_partial_der(w0, w1, w1)))

    # showPlot()
    error_vs_epochs_data, loss_vs_epochs_data = gradientDescentAlg()
    # print(error_vs_epochs_data)
    epochs_indexes = range(len(error_vs_epochs_data))
    plt.scatter(epochs_indexes, error_vs_epochs_data)
    plt.scatter(epochs_indexes, loss_vs_epochs_data)
    plt.show()
