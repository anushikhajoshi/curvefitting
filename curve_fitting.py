#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:26:16 2022

@author: anushikhajoshi
"""

# Reading and Plotting the Data
import numpy as np
import matplotlib.pyplot as plt
rocket = np.loadtxt("rocket.csv", delimiter = ",")

plt.figure(figsize = (20, 10))
plt.errorbar(rocket[:,0], rocket[:,1], yerr= rocket[:,2], fmt= 'o')
plt.xlabel('time(h)')
plt.ylabel('position(km)')
plt.title('position-time graph of rocket with error')
plt.savefig('position-time graph of rocket with error.png')
plt.show()


 #  Estimating the speed
def estimating_speed(data): 
    speed = data[1:,1]/data[1:,0]
    return (np.mean(speed), np.std(speed))

speed_mean, speed_std = estimating_speed(rocket)
print('Mean for speed: ', speed_mean)
print('Standard Deviation for speed: ', speed_std)


#  Linear Regression
def linear_regression_model(data):
    # numerator 
    nu = (data[:,0] - np.mean(data[:,0])) * (data[:,1] - np.mean(data[:,1]))
    # denominator
    de = ((data[:,0] - np.mean(data[:,0])))**2
    # estimated mean
    u = nu.sum()/de.sum()
    # estimated distance
    d0 = (np.mean(data[:,1])) - (u * (np.mean(data[:,0])))                        
    return d0, u 

d0, u = linear_regression_model(rocket)
print('Estimated initial position: ', d0)
print('Estimated speed: ', u)


# Plotting the prediction
def get_rocket_prediction(time, position, speed):
    distance = position + speed*time
    return distance

predicted_position = get_rocket_prediction(rocket[:,0], d0, u)

plt.figure(figsize = (20, 10))
plt.errorbar(rocket[:,0], rocket[:,1], yerr= rocket[:,2], fmt= 'o', label = 'measured data')
plt.plot(rocket[:,0], predicted_position, label = 'predicted data')
plt.xlabel('time(h)')
plt.ylabel('position(km)')
plt.title('measured and predicted data')
plt.legend()
plt.savefig('measured and predicted data.png')
plt.show()


# Characterizing the fit
def characterized_fit(data, prediction):
    error = (data[:,1] - prediction)**2
    si = (data[:,2])**2
    chi_square = (1/(len(data) - 2))*(np.sum(error)/np.sum(si))
    return chi_square

print('Chi Square for linear fit: ', characterized_fit(rocket, predicted_position))
        


# Curve fitting

from scipy.optimize import curve_fit

popt, pcov = curve_fit(get_rocket_prediction, 
                       xdata=rocket[:,0], 
                       ydata=rocket[:,1], 
                       sigma=rocket[:,2], 
                       absolute_sigma=True, 
                       p0=(0, 0))

pstd = np.sqrt(np.diag(pcov))

# Part 1: Print out the estimates and their associated uncertainties (standard deviations).
print('New estimate for position is ', popt[0] , ' km and its standard deviation is ', pstd[0])
print('New estimate for speed is ', popt[1] , ' km/h and its standard deviation is ', pstd[1])

# Part 2: Print out the chi-square for the model with the new best estimate parameters.
new_prediction = get_rocket_prediction(rocket[:,0], popt[0], popt[1])
curve_chi = characterized_fit(rocket, new_prediction)
print('Chi Square for curve fit: ', curve_chi)

# Part 3: Plot the data with uncertainties, and plot the model using your linear regression parameters and the curve_fit parameters as separate lines.
plt.figure(figsize = (20, 10))
plt.errorbar(rocket[:,0], rocket[:,1], yerr= rocket[:,2], fmt= 'o', label = 'measured data')
plt.plot(rocket[:,0], predicted_position, label = 'predicted data - linear')
plt.plot(rocket[:,0], new_prediction, label = 'predicted data - curve')
plt.xlabel('time(h)')
plt.ylabel('position(km)')
plt.title('measured and predicted data')
plt.legend()
plt.savefig('measured and predicted data with curve fit.png')
plt.show()


# Load Data
feather = np.loadtxt("feather .csv", delimiter = ",")

# Predict the position of the feather above the Lunar surface as a function of time given the initial position s0, 
# initial speed u, and constant acceleration a.

def predict_feather_position(time, position, speed, acceleration): 
    distance = position + (speed*time) + (0.5*acceleration*(time)**2)
    return distance

# Step 3: Fit the model to the data to find the optimal (i.e. best) parameters
popt, pcov = curve_fit(predict_feather_position, 
                       xdata=feather[:,0], 
                       ydata=feather[:,1], 
                       sigma=feather[:,2], 
                       absolute_sigma=True, 
                       p0=(0, 0, 0))

pstd = np.sqrt(np.diag(pcov))

# Print out the parameters fit by your model, including their uncertainties and units 
print('New estimate for position is ', popt[0] , ' m and its standard deviation is ', pstd[0])
print('New estimate for speed is ', popt[1] , ' m/s and its standard deviation is ', pstd[1])
print('New estimate for acceleration is ', popt[2] , ' m/s^2 and its standard deviation is ', pstd[2])

feather_prediction = predict_feather_position(feather[:,0], popt[0], popt[1], popt[2])

# Plot the data including errorbars, and a prediction made by your model function using the best parameters

plt.figure(figsize = (20, 10))
plt.errorbar(feather[:,0], feather[:,1], yerr= feather[:,2], fmt= 'o', label = 'measured data')
plt.plot(feather[:,0], feather_prediction, label = 'predicted data')
plt.xlabel('time(h)')
plt.ylabel('position(km)')
plt.title('measured and predicted data')
plt.legend()
plt.savefig('measured and predicted data for feather.png')
plt.show()





    

    

    
    
