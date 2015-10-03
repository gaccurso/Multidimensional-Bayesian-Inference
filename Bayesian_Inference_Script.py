from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import sys
import emcee
from multiprocessing import Pool
import triangle
import pylab
import os
import itertools
from num2words import num2words

###########New Bayesian functions to fit for one parameters everything###############

def model_one(theta, x1):
    return theta[0]*x1 + theta[1]

def log_pr(theta):
    if any(t > 1.0e+10 for t in theta):
        return -np.inf
    if any(t < -1.0e+10 for t in theta):
        return -np.inf
    return 0.0

def log_likelihood_simple_one(theta, x1, y, e, x1error, sample_size):
    y_model = model_one(theta, x1)
    y_model_sigma = 0.0   #np.var(y)
    error_terms = (theta[0]*x1error)**2.0
    return -((sample_size/2.0)*(np.log(2.0*np.pi))) - 0.5*np.sum(np.log((y_model_sigma+(e**2.0)+error_terms))) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    
    ####return -((sample_size/2.0)*(np.log(2.0*np.pi))) + 0.5*np.sum(np.log(((np.sum(theta[:-1]**2.0))+1.0)/(y_model_sigma+(e**2.0)+error_terms)) ) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms))     ###      
    ###return -2.0*np.sum(((y - y_model) ** 2.0)/((2.0*e) ** 2.0)) - (sample_size/2.0)*(1.837877) - np.sum(np.log(e))
    
def chi_squared_one(theta, x1, y, e):
    y_model = model_one(theta, x1)
    return np.sum(((y - y_model) ** 2.0) / (e ** 2.0))
    
def pearsons_chi_squared_one(theta, x1, y):
    y_model = model_one(theta, x1)
    return np.sum(((y - y_model)**2.0)/(y_model))

def log_post_one(theta, x1, y, e, x1error, sample_size):
    if not np.isfinite(log_pr(theta)):
        return -np.inf
    return log_pr(theta) + log_likelihood_simple_one(theta, x1, y, e, x1error, sample_size)

###########New Bayesian functions to fit for two parameters everything###############

def model_two(theta, x1, x2):
    return theta[0]*x1 + theta[1]*x2 + theta[2]

def log_pr(theta):
    if any(t > 1.0e+10 for t in theta):
        return -np.inf
    if any(t < -1.0e+10 for t in theta):
        return -np.inf
    return 0.0


def log_likelihood_simple_two(theta, x1, x2, y, e, x1error, x2error, sample_size):
    y_model = model_two(theta, x1, x2)
    y_model_sigma = 0.0   #np.var(y)
    error_terms = ((theta[0]*x1error)**2.0) + ((theta[1]*x2error)**2.0)
    return -((sample_size/2.0)*(np.log(2.0*np.pi))) - 0.5*np.sum(np.log((y_model_sigma+(e**2.0)+error_terms))) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    ###return -((sample_size/2.0)*(np.log(2.0*np.pi))) + 0.5*np.sum(np.log(((np.sum(theta[:-1]**2.0))+1.0)/(y_model_sigma+(e**2.0)+error_terms)) ) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms))  
    

def chi_squared_two(theta, x1, x2, y, e):
	y_model = model_two(theta, x1, x2)
	return np.sum(((y - y_model) ** 2.0) / (e ** 2.0))
	
def pearsons_chi_squared_two(theta, x1, x2, y):
	y_model = model_two(theta, x1, x2)
	return np.sum(((y - y_model)**2.0)/(y_model))
	
def log_post_two(theta, x1, x2, y, e, x1error, x2error, sample_size):
    if not np.isfinite(log_pr(theta)):
        return -np.inf
    return log_pr(theta) + log_likelihood_simple_two(theta, x1, x2, y, e, x1error, x2error, sample_size)

###########New Bayesian functions to fit for three parameters everything###############


def model_three(theta, x1, x2, x3):
    return theta[0]*x1 + theta[1]*x2 + theta[2]*x3 + theta[3]

def log_pr(theta):
    if any(t > 1.0e+10 for t in theta):
        return -np.inf
    if any(t < -1.0e+10 for t in theta):
        return -np.inf
    return 0.0

def log_likelihood_simple_three(theta, x1, x2, x3, y, e, x1error, x2error, x3error, sample_size):
    y_model = model_three(theta, x1, x2, x3)
    y_model_sigma = 0.0   #np.var(y)
    error_terms = ((theta[0]*x1error)**2.0) + ((theta[1]*x2error)**2.0)+ ((theta[2]*x3error)**2.0)
    return -((sample_size/2.0)*(np.log(2.0*np.pi))) - 0.5*np.sum(np.log((y_model_sigma+(e**2.0)+error_terms))) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    ###return -((sample_size/2.0)*(np.log(2.0*np.pi))) + 0.5*np.sum(np.log(((np.sum(theta[:-1]**2.0))+1.0)/(y_model_sigma+(e**2.0)+error_terms)) ) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms))  
    
    
def chi_squared_three(theta, x1, x2, x3, y, e):
	y_model = model_three(theta, x1, x2, x3)
	return np.sum(((y - y_model) ** 2.0) / (e ** 2.0))
	
def pearsons_chi_squared_three(theta, x1, x2, x3, y):
	y_model = model_three(theta, x1, x2, x3)
	return np.sum(((y - y_model)**2.0)/(y_model))
	
def log_post_three(theta, x1, x2, x3, y, e, x1error, x2error, x3error, sample_size):
    if not np.isfinite(log_pr(theta)):
        return -np.inf
    return log_pr(theta) + log_likelihood_simple_three(theta, x1, x2, x3, y, e,  x1error, x2error, x3error, sample_size)


###########New Bayesian functions to fit for four parameters everything###############

def model_four(theta, x1, x2, x3, x4):
    return theta[0]*x1 + theta[1]*x2 + theta[2]*x3 + theta[3]*x4 + theta[4]

def log_pr(theta):
    if any(t > 1.0e+10 for t in theta):
        return -np.inf
    if any(t < -1.0e+10 for t in theta):
        return -np.inf
    return 0.0

def log_likelihood_simple_four(theta, x1, x2, x3, x4, y, e, x1error, x2error, x3error, x4error, sample_size):
    y_model = model_four(theta, x1, x2, x3, x4)
    y_model_sigma = 0.0   #np.var(y)
    error_terms = ((theta[0]*x1error)**2.0) + ((theta[1]*x2error)**2.0)+ ((theta[2]*x3error)**2.0) + ((theta[3]*x4error)**2.0)
    return -((sample_size/2.0)*(np.log(2.0*np.pi))) - 0.5*np.sum(np.log((y_model_sigma+(e**2.0)+error_terms))) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    #return -((sample_size/2.0)*(np.log(2.0*np.pi))) + 0.5*np.sum(np.log(((np.sum(theta[:-1]**2.0))+1.0)/(y_model_sigma+(e**2.0)+error_terms)) ) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    
def chi_squared_four(theta, x1, x2, x3, x4, y, e):
	y_model = model_four(theta, x1, x2, x3, x4)
	return np.sum(((y - y_model) ** 2.0) / (e ** 2.0))
	
def pearsons_chi_squared_four(theta, x1, x2, x3, x4, y):
	y_model = model_four(theta, x1, x2, x3, x4)
	return np.sum(((y - y_model)**2.0)/(y_model))
	
def log_post_four(theta, x1, x2, x3, x4, y, e, x1error, x2error, x3error, x4error, sample_size):
    if not np.isfinite(log_pr(theta)):
        return -np.inf
    return log_pr(theta) + log_likelihood_simple_four(theta, x1, x2, x3, x4, y, e, x1error, x2error, x3error, x4error, sample_size)

###########New Bayesian functions to fit for five parameters everything###############
def model_five(theta, x1, x2, x3, x4, x5):
    return theta[0]*x1 + theta[1]*x2 + theta[2]*x3 + theta[3]*x4 + theta[4]*x5 + theta[5]

def log_pr(theta):
    if any(t > 1.0e+10 for t in theta):
        return -np.inf
    if any(t < -1.0e+10 for t in theta):
        return -np.inf
    return 0.0
    
def log_likelihood_simple_five(theta, x1, x2, x3, x4, x5, y, e, x1error, x2error, x3error, x4error, x5error, sample_size):
    y_model = model_five(theta, x1, x2, x3, x4, x5)
    y_model_sigma = 0.0   #np.var(y)
    error_terms = ((theta[0]*x1error)**2.0) + ((theta[1]*x2error)**2.0)+ ((theta[2]*x3error)**2.0) + ((theta[3]*x4error)**2.0) + ((theta[4]*x5error)**2.0)
    return -((sample_size/2.0)*(np.log(2.0*np.pi))) - 0.5*np.sum(np.log((y_model_sigma+(e**2.0)+error_terms))) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    #return -((sample_size/2.0)*(np.log(2.0*np.pi))) + 0.5*np.sum(np.log(((np.sum(theta[:-1]**2.0))+1.0)/(y_model_sigma+(e**2.0)+error_terms)) ) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    
def chi_squared_five(theta, x1, x2, x3, x4, x5, y, e):
	y_model = model_five(theta, x1, x2, x3, x4, x5)
	return np.sum(((y - y_model) ** 2.0) / (e ** 2.0))
	
def pearsons_chi_squared_five(theta, x1, x2, x3, x4, x5, y):
	y_model = model_five(theta, x1, x2, x3, x4, x5)
	return np.sum(((y - y_model)**2.0)/(y_model))
	
def log_post_five(theta, x1, x2, x3, x4, x5, y, e, x1error, x2error, x3error, x4error, x5error, sample_size):
    if not np.isfinite(log_pr(theta)):
        return -np.inf
    return log_pr(theta) + log_likelihood_simple_five(theta, x1, x2, x3, x4, x5, y, e, x1error, x2error, x3error, x4error, x5error, sample_size)
    
###########New Bayesian functions to fit for six parameters everything###############

def model_six(theta, x1, x2, x3, x4, x5, x6):
    return theta[0]*x1 + theta[1]*x2 + theta[2]*x3 + theta[3]*x4 + theta[4]*x5 + theta[5]*x6 + theta[6]

def log_pr(theta):
    if any(t > 1.0e+10 for t in theta):
        return -np.inf
    if any(t < -1.0e+10 for t in theta):
        return -np.inf
    return 0.0

def log_likelihood_simple_six(theta, x1, x2, x3, x4, x5, x6, y, e, x1error, x2error, x3error, x4error, x5error, x6error, sample_size):
    y_model = model_six(theta, x1, x2, x3, x4, x5, x6)
    y_model_sigma = 0.0   #np.var(y)
    error_terms = ((theta[0]*x1error)**2.0) + ((theta[1]*x2error)**2.0)+ ((theta[2]*x3error)**2.0) + ((theta[3]*x4error)**2.0) + ((theta[4]*x5error)**2.0) + ((theta[5]*x6error)**2.0)
    return -((sample_size/2.0)*(np.log(2.0*np.pi))) - 0.5*np.sum(np.log((y_model_sigma+(e**2.0)+error_terms))) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    #return -((sample_size/2.0)*(np.log(2.0*np.pi))) + 0.5*np.sum(np.log(((np.sum(theta[:-1]**2.0))+1.0)/(y_model_sigma+(e**2.0)+error_terms)) ) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    
def chi_squared_six(theta, x1, x2, x3, x4, x5, x6, y, e):
	y_model = model_six(theta, x1, x2, x3, x4, x5, x6)
	return np.sum(((y - y_model) ** 2.0) / (e ** 2.0))
	
def pearsons_chi_squared_six(theta, x1, x2, x3, x4, x5, x6, y):
	y_model = model_six(theta, x1, x2, x3, x4, x5, x6)
	return np.sum(((y - y_model)**2.0)/(y_model))

def log_post_six(theta, x1, x2, x3, x4, x5, x6, y, e, x1error, x2error, x3error, x4error, x5error, x6error, sample_size):
    if not np.isfinite(log_pr(theta)):
        return -np.inf
    return log_pr(theta) + log_likelihood_simple_six(theta, x1, x2, x3, x4, x5, x6, y, e, x1error, x2error, x3error, x4error, x5error, x6error, sample_size)
    

###########New Bayesian functions to fit for seven parameters everything###############
def model_seven(theta, x1, x2, x3, x4, x5, x6, x7):
    return theta[0]*x1 + theta[1]*x2 + theta[2]*x3 + theta[3]*x4 + theta[4]*x5 + theta[5]*x6 + theta[6]*x7 + theta[7]

def log_pr(theta):
    if any(t > 1.0e+10 for t in theta):
        return -np.inf
    if any(t < -1.0e+10 for t in theta):
        return -np.inf
    return 0.0

def log_likelihood_simple_seven(theta, x1, x2, x3, x4, x5, x6, x7, y, e, x1error, x2error, x3error, x4error, x5error, x6error, x7error, sample_size):
    y_model = model_seven(theta, x1, x2, x3, x4, x5, x6, x7)
    y_model_sigma = 0.0   #np.var(y)
    error_terms = ((theta[0]*x1error)**2.0) + ((theta[1]*x2error)**2.0)+ ((theta[2]*x3error)**2.0) + ((theta[3]*x4error)**2.0) + ((theta[4]*x5error)**2.0) + ((theta[5]*x6error)**2.0) + ((theta[6]*x7error)**2.0)
    return -((sample_size/2.0)*(np.log(2.0*np.pi))) - 0.5*np.sum(np.log((y_model_sigma+(e**2.0)+error_terms))) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    #return -((sample_size/2.0)*(np.log(2.0*np.pi))) + 0.5*np.sum(np.log(((np.sum(theta[:-1]**2.0))+1.0)/(y_model_sigma+(e**2.0)+error_terms)) ) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 

def chi_squared_seven(theta, x1, x2, x3, x4, x5, x6, x7, y, e):
	y_model = model_seven(theta, x1, x2, x3, x4, x5, x6, x7)
	return np.sum(((y - y_model) ** 2.0) / (e ** 2.0))
	
def pearsons_chi_squared_seven(theta, x1, x2, x3, x4, x5, x6, x7, y):
	y_model = model_seven(theta, x1, x2, x3, x4, x5, x6, x7)
	return np.sum(((y - y_model)**2.0)/(y_model))
	
def log_post_seven(theta, x1, x2, x3, x4, x5, x6, x7, y, e, x1error, x2error, x3error, x4error, x5error, x6error, x7error, sample_size):
    if not np.isfinite(log_pr(theta)):
        return -np.inf
    return log_pr(theta) + log_likelihood_simple_seven(theta, x1, x2, x3, x4, x5, x6, x7, y, e, x1error, x2error, x3error, x4error, x5error, x6error, x7error, sample_size)

###########New Bayesian functions to fit for eight parameters everything###############

def model_eight(theta, x1, x2, x3, x4, x5, x6, x7, x8):
    return theta[0]*x1 + theta[1]*x2 + theta[2]*x3 + theta[3]*x4 + theta[4]*x5 + theta[5]*x6 + theta[6]*x7 + theta[7]*x8 + theta[8]

def log_pr(theta):
    if any(t > 1.0e+10 for t in theta):
        return -np.inf
    if any(t < -1.0e+10 for t in theta):
        return -np.inf
    return 0.0

def log_likelihood_simple_eight(theta, x1, x2, x3, x4, x5, x6, x7, x8, y, e, x1error, x2error, x3error, x4error, x5error, x6error, x7error, x8error, sample_size):
    y_model = model_eight(theta, x1, x2, x3, x4, x5, x6, x7, x8)
    y_model_sigma = 0.0   #np.var(y)
    error_terms = ((theta[0]*x1error)**2.0) + ((theta[1]*x2error)**2.0)+ ((theta[2]*x3error)**2.0) + ((theta[3]*x4error)**2.0) + ((theta[4]*x5error)**2.0) + ((theta[5]*x6error)**2.0) + ((theta[6]*x7error)**2.0) + ((theta[7]*x8error)**2.0)
    return -((sample_size/2.0)*(np.log(2.0*np.pi))) - 0.5*np.sum(np.log((y_model_sigma+(e**2.0)+error_terms))) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    #return -((sample_size/2.0)*(np.log(2.0*np.pi))) + 0.5*np.sum(np.log(((np.sum(theta[:-1]**2.0))+1.0)/(y_model_sigma+(e**2.0)+error_terms)) ) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 

def chi_squared_eight(theta, x1, x2, x3, x4, x5, x6, x7, x8, y, e):
	y_model = model_eight(theta, x1, x2, x3, x4, x5, x6, x7, x8)
	return np.sum(((y - y_model) ** 2.0) / (e ** 2.0))
	
def pearsons_chi_squared_eight(theta, x1, x2, x3, x4, x5, x6, x7, x8, y):
	y_model = model_eight(theta, x1, x2, x3, x4, x5, x6, x7, x8)
	return np.sum(((y - y_model)**2.0)/(y_model))
	
def log_post_eight(theta, x1, x2, x3, x4, x5, x6, x7, x8, y, e, x1error, x2error, x3error, x4error, x5error, x6error, x7error, x8error, sample_size):
    if not np.isfinite(log_pr(theta)):
        return -np.inf
    return log_pr(theta) + log_likelihood_simple_eight(theta, x1, x2, x3, x4, x5, x6, x7, x8, y, e, x1error, x2error, x3error, x4error, x5error, x6error, x7error, x8error, sample_size)

###########New Bayesian functions to fit for nine parameters everything###############
def model_nine(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return theta[0]*x1 + theta[1]*x2 + theta[2]*x3 + theta[3]*x4 + theta[4]*x5 + theta[5]*x6 + theta[6]*x7 + theta[7]*x8 + theta[8]*x9 + theta[9]

def log_pr(theta):
    if any(t > 1.0e+10 for t in theta):
        return -np.inf
    if any(t < -1.0e+10 for t in theta):
        return -np.inf
    return 0.0
    
def log_likelihood_simple_nine(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, y, e, x1error, x2error, x3error, x4error, x5error, x6error, x7error, x8error, x9error, sample_size):
    y_model = model_nine(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    y_model_sigma = 0.0   #np.var(y)
    error_terms = ((theta[0]*x1error)**2.0) + ((theta[1]*x2error)**2.0)+ ((theta[2]*x3error)**2.0) + ((theta[3]*x4error)**2.0) + ((theta[4]*x5error)**2.0) + ((theta[5]*x6error)**2.0) + ((theta[6]*x7error)**2.0) + ((theta[7]*x8error)**2.0) + ((theta[8]*x9error)**2.0)
    return -((sample_size/2.0)*(np.log(2.0*np.pi))) - 0.5*np.sum(np.log((y_model_sigma+(e**2.0)+error_terms))) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 
    #return -((sample_size/2.0)*(np.log(2.0*np.pi))) + 0.5*np.sum(np.log(((np.sum(theta[:-1]**2.0))+1.0)/(y_model_sigma+(e**2.0)+error_terms)) ) - 0.5*np.sum(((y-y_model)**2.0)/(y_model_sigma+(e**2.0)+error_terms)) 

def chi_squared_nine(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, y, e):
	y_model = model_nine(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9)
	return np.sum(((y - y_model) ** 2.0) / (e ** 2.0))
	
def pearsons_chi_squared_nine(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, y):
	y_model = model_nine(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9)
	return np.sum(((y - y_model)**2.0)/(y_model))
	
def log_post_nine(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, y, e, x1error, x2error, x3error, x4error, x5error, x6error, x7error, x8error, x9error, sample_size):
    if not np.isfinite(log_pr(theta)):
        return -np.inf
    return log_pr(theta) + log_likelihood_simple_nine(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, y, e, x1error, x2error, x3error, x4error, x5error, x6error, x7error, x8error, x9error, sample_size)

###############################Now to do the Bayesian fit for one parameter#################################
np.random.seed(0)
cores = 8
nparameters = 1
ndim = nparameters + 1				# this should be the number of parameters + 1.
nwalkers = 200
nburn = 500
nsteps = 16000
print('Number of free parameters in our model =', ndim)
headings = ['Stellar Mass', 'SFR', 'Colour', 'Mu', 'sSFR', 'Metallicity', 'Hardness', 'Extinction']	
datatouse = np.loadtxt("Python_herschel_analysis.dat")
f = open('herschel_stats_onedim_data.txt', 'w')
f.close()

for i in range(len(list(itertools.combinations('12345678', 1)))):
	x = list(itertools.combinations('12345678', 1))				#one parameter hence why repeat equals 1
	y = list(x[i])
	print(y[0])														#number of numbers in the combinatorix - 1 	for the first etc										
	ydata = datatouse[:,0]
	ydataerror = datatouse[:,9]
	x1data = datatouse[:,y[0]]
	print(y[0])
	x1dataerror = datatouse[:,float(y[0])+9.0]
	indicestoignore = np.hstack([np.where(np.isnan(ydata)), np.where(np.isinf(ydata)), np.where(np.isnan(x1data)), np.where(np.isinf(x1data)), np.where(np.isnan(ydataerror)), np.where(np.isinf(ydataerror))])
	print(indicestoignore)
	datatousenew = np.delete(datatouse, (indicestoignore), axis=0)

	ydata = datatousenew[:,0]
	ydataerror = datatousenew[:,9]
	x1data = datatousenew[:,y[0]]
	x1dataerror = datatousenew[:,float(y[0])+9.0]

	starting_guesses = np.random.rand(nwalkers, ndim)
	pool = Pool(cores)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_one, args=[x1data, ydata, ydataerror, x1dataerror, len(x1data)], pool=pool)
	sampler.run_mcmc(starting_guesses, nsteps)
	sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
	sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
	
	logposteriors = sampler.lnprobability[:, nburn:].reshape(-1)
	answers = np.mean(sample[np.where(logposteriors== np.max(logposteriors))],0).reshape(ndim)
	theta3 = answers
	
	figure = triangle.corner(sample, labels=[headings[int(y[0])-1], 'Intercept'], truths=theta3)
	figure.savefig("triangle_one_dim_test_"+str(headings[int(y[0])-1])+".png", format='png')
	
	print(theta3)
	print(chi_squared_one(theta3, x1data, ydata, ydataerror))
	print(pearsons_chi_squared_one(theta3, x1data, ydata))
	print(log_likelihood_simple_one(theta3, x1data, ydata, ydataerror, x1dataerror, len(x1data)))
	print(len(x1data))

	f = open('herschel_stats_onedim_data.txt', 'a')
	f.write(str(theta3[0]))
	f.write(str("    "))
	f.write(str(theta3[1]))
	f.write(str("    "))
	f.write(str(chi_squared_one(theta3, x1data, ydata, ydataerror)))
	f.write(str("    "))
	f.write(str(pearsons_chi_squared_one(theta3, x1data, ydata)))
	f.write(str("    "))
	f.write(str(log_likelihood_simple_one(theta3, x1data, ydata, ydataerror,  x1dataerror,  len(x1data)))) #  -(len(set(indicestoignore[0]))/2.0)*(1.837877)-np.sum(np.log(datatouse[list(set(indicestoignore[0])),10]))))
	f.write(str("    "))
	f.write(str(len(x1data)))
	f.write(str("    "))
	f.write(str(len(indicestoignore[0])))
	f.write(str("    "))
	f.write(str(len(set(indicestoignore[0]))))
	f.write("\n")
	f.close()
	
	sampler.pool.terminate()
	
	plt.clf()
	plt.close()

	del theta3
	del ydata
	del ydataerror
	del x1data
	del sample
	del sampler
	del figure
	

###############################Now to do the Bayesian fit for two parameters#################################
np.random.seed(0)
cores = 8
nparameters = 2
ndim = nparameters + 1				# this should be the number of parameters + 1.
nwalkers = 200
nburn = 500
nsteps = 16000
print('Number of free parameters in our model =', ndim)
datatouse = np.loadtxt("Python_herschel_analysis.dat")
f = open('herschel_stats_twodim_data.txt', 'w')
f.close()

for i in range(len(list(itertools.combinations('12345678', 2)))):
	x = list(itertools.combinations('12345678', 2))				#two parameter hence why repeat equals 1
	y = list(x[i])
	print(y[0])	
	print(y[1])													#number of numbers in the combinatorix - 1 	for the first etc										
	ydata = datatouse[:,0]
	ydataerror = datatouse[:,9]
	x1data = datatouse[:,y[0]]
	x2data = datatouse[:,y[1]]
	x1dataerror = datatouse[:,float(y[0])+9.0]
	x2dataerror = datatouse[:,float(y[1])+9.0]
	indicestoignore = np.hstack([np.where(np.isnan(ydata)), np.where(np.isinf(ydata)), np.where(np.isnan(x1data)), np.where(np.isinf(x1data)), np.where(np.isnan(x2data)), np.where(np.isinf(x2data)), np.where(np.isnan(ydataerror)), np.where(np.isinf(ydataerror))])
	print(indicestoignore)
	datatousenew = np.delete(datatouse, (indicestoignore), axis=0)

	ydata = datatousenew[:,0]
	ydataerror = datatousenew[:,9]
	x1data = datatousenew[:,y[0]]
	x2data = datatousenew[:,y[1]]
	x1dataerror = datatousenew[:,float(y[0])+9.0]
	x2dataerror = datatousenew[:,float(y[1])+9.0]
	
	starting_guesses = np.random.rand(nwalkers, ndim)
	pool = Pool(cores)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_two, args=[x1data, x2data, ydata, ydataerror, x1dataerror, x2dataerror, len(x1data)], pool=pool)
	sampler.run_mcmc(starting_guesses, nsteps)
	sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
	sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
	
	logposteriors = sampler.lnprobability[:, nburn:].reshape(-1)
	answers = np.mean(sample[np.where(logposteriors== np.max(logposteriors))],0).reshape(ndim)
	theta3 = answers
	
	figure = triangle.corner(sample, labels=[ headings[int(y[0])-1], headings[int(y[1])-1], 'Intercept'], truths=theta3)
	figure.savefig("triangle_two_dim_test_"+str(headings[int(y[0])-1])+"_"+str(headings[int(y[1])-1])+".png", format='png')

	print(theta3)
	print(chi_squared_two(theta3, x1data, x2data, ydata, ydataerror))
	print(pearsons_chi_squared_two(theta3, x1data, x2data, ydata))
	print(log_likelihood_simple_two(theta3, x1data, x2data, ydata, ydataerror,  x1dataerror, x2dataerror,  len(x1data)))
	print(len(x1data))
	
	f = open('herschel_stats_twodim_data.txt', 'a')
	f.write(str(theta3[0]))
	f.write(str("    "))
	f.write(str(theta3[1]))
	f.write(str("    "))
	f.write(str(theta3[2]))
	f.write(str("    "))
	f.write(str(chi_squared_two(theta3, x1data, x2data, ydata, ydataerror)))
	f.write(str("    "))
	f.write(str(pearsons_chi_squared_two(theta3, x1data, x2data, ydata)))
	f.write(str("    "))
	f.write(str(log_likelihood_simple_two(theta3, x1data, x2data, ydata, ydataerror,  x1dataerror, x2dataerror, len(x1data)))) #-(len(set(indicestoignore[0]))/2.0)*(1.837877)-np.sum(np.log(datatouse[list(set(indicestoignore[0])),10]))))
	f.write(str("    "))
	f.write(str(len(x1data)))
	f.write(str("    "))
	f.write(str(len(indicestoignore[0])))
	f.write(str("    "))
	f.write(str(len(set(indicestoignore[0]))))
	f.write("\n")
	f.close()

	sampler.pool.terminate()
	
	plt.clf()
	plt.close()
	
	del theta3
	del ydata
	del ydataerror
	del x1data
	del x2data
	del sample
	del sampler
	del figure
	
	
	

###############################Now to do the Bayesian fit for three parameters#################################
np.random.seed(0)
cores = 8
nparameters = 3
ndim = nparameters + 1				# this should be the number of parameters + 1.
nwalkers = 200
nburn = 500
nsteps = 16000
print('Number of free parameters in our model =', ndim)
datatouse = np.loadtxt("Python_herschel_analysis.dat")
f = open('herschel_stats_threedim_data.txt', 'w')
f.close()


for i in range(len(list(itertools.combinations('12345678', 3)))):
	x = list(itertools.combinations('12345678', 3))				#three parameter hence why repeat equals 1
	y = list(x[i])
	print(y[0])	
	print(y[1])													#number of numbers in the combinatorix - 1 	for the first etc										
	print(y[2])
	
	
	ydata = datatouse[:,0]
	ydataerror = datatouse[:,9]
	x1data = datatouse[:,y[0]]
	x2data = datatouse[:,y[1]]
	x3data = datatouse[:,y[2]]
	x1dataerror = datatouse[:,float(y[0])+9.0]
	x2dataerror = datatouse[:,float(y[1])+9.0]
	x3dataerror = datatouse[:,float(y[2])+9.0]
	indicestoignore = np.hstack([np.where(np.isnan(ydata)), np.where(np.isinf(ydata)), np.where(np.isnan(x3data)), np.where(np.isinf(x3data)), np.where(np.isnan(x1data)), np.where(np.isinf(x1data)), np.where(np.isnan(x2data)), np.where(np.isinf(x2data)), np.where(np.isnan(ydataerror)), np.where(np.isinf(ydataerror))])
	print(indicestoignore)
	datatousenew = np.delete(datatouse, (indicestoignore), axis=0)

	ydata = datatousenew[:,0]
	ydataerror = datatousenew[:,9]
	x1data = datatousenew[:,y[0]]
	x2data = datatousenew[:,y[1]]
	x3data = datatousenew[:,y[2]]
	x1dataerror = datatousenew[:,float(y[0])+9.0]
	x2dataerror = datatousenew[:,float(y[1])+9.0]
	x3dataerror = datatousenew[:,float(y[2])+9.0]

	starting_guesses = np.random.rand(nwalkers, ndim)
	pool = Pool(cores)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_three, args=[x1data, x2data, x3data, ydata, ydataerror, x1dataerror, x2dataerror, x3dataerror, len(x1data)], pool=pool)
	sampler.run_mcmc(starting_guesses, nsteps)
	sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
	sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
	
	logposteriors = sampler.lnprobability[:, nburn:].reshape(-1)
	answers = np.mean(sample[np.where(logposteriors== np.max(logposteriors))],0).reshape(ndim)
	theta3 = answers
	
	figure = triangle.corner(sample, labels=[ headings[int(y[0])-1], headings[int(y[1])-1], headings[int(y[2])-1], 'Intercept'], truths=theta3)
	figure.savefig("triangle_three_dim_test_"+str(headings[int(y[0])-1])+"_"+str(headings[int(y[1])-1])+"_"+str(headings[int(y[2])-1])+".png", format='png')


	print(chi_squared_three(theta3, x1data, x2data, x3data, ydata, ydataerror))
	print(pearsons_chi_squared_three(theta3, x1data, x2data, x3data, ydata))
	print(log_likelihood_simple_three(theta3, x1data, x2data, x3data, ydata, ydataerror,x1dataerror, x2dataerror, x3dataerror, len(x1data)))
	print(len(x1data))
	
	f = open('herschel_stats_threedim_data.txt', 'a')
	f.write(str(theta3[0]))
	f.write(str("    "))
	f.write(str(theta3[1]))
	f.write(str("    "))
	f.write(str(theta3[2]))
	f.write(str("    "))
	f.write(str(theta3[3]))
	f.write(str("    "))
	f.write(str(chi_squared_three(theta3, x1data, x2data, x3data, ydata, ydataerror)))
	f.write(str("    "))
	f.write(str(pearsons_chi_squared_three(theta3, x1data, x2data, x3data, ydata)))
	f.write(str("    "))
	f.write(str(log_likelihood_simple_three(theta3, x1data, x2data, x3data, ydata, ydataerror, x1dataerror, x2dataerror, x3dataerror, len(x1data)))) #-(len(set(indicestoignore[0]))/2.0)*(1.837877)-np.sum(np.log(datatouse[list(set(indicestoignore[0])),10]))))
	f.write(str("    "))
	f.write(str(len(x1data)))
	f.write(str("    "))
	f.write(str(len(indicestoignore[0])))
	f.write(str("    "))
	f.write(str(len(set(indicestoignore[0]))))
	f.write("\n")
	f.close()

	sampler.pool.terminate()
	
	plt.clf()
	plt.close()			#new things we tried 

	del x
	del y
	del theta3
	del ydata
	del ydataerror
	del x1data
	del x2data
	del x3data
	del sample
	del sampler
	del figure
	

###############################Now to do the Bayesian fit for four parameters#################################
np.random.seed(0)
cores = 8
nparameters = 4
ndim = nparameters + 1				# this should be the number of parameters + 1.
nwalkers = 200
nburn = 500
nsteps = 16000
print('Number of free parameters in our model =', ndim)
datatouse = np.loadtxt("Python_herschel_analysis.dat")
f = open('herschel_stats_fourdim_data.txt', 'w')
f.close()
	


for i in range(len(list(itertools.combinations('12345678', 4)))):
	x = list(itertools.combinations('12345678', 4))				#three parameter hence why repeat equals 1
	y = list(x[i])
	print(y[0])	
	print(y[1])													#number of numbers in the combinatorix - 1 	for the first etc										
	print(y[2])
	print(y[3])
	
	
	ydata = datatouse[:,0]
	ydataerror = datatouse[:,9]
	x1data = datatouse[:,y[0]]
	x2data = datatouse[:,y[1]]
	x3data = datatouse[:,y[2]]
	x4data = datatouse[:,y[3]]
	x1dataerror = datatouse[:,float(y[0])+9.0]
	x2dataerror = datatouse[:,float(y[1])+9.0]
	x3dataerror = datatouse[:,float(y[2])+9.0]
	x4dataerror = datatouse[:,float(y[3])+9.0]
	indicestoignore = np.hstack([np.where(np.isnan(ydata)), np.where(np.isinf(ydata)), np.where(np.isnan(x4data)), np.where(np.isinf(x4data)),  np.where(np.isnan(x3data)), np.where(np.isinf(x3data)), np.where(np.isnan(x1data)), np.where(np.isinf(x1data)),  np.where(np.isnan(x2data)), np.where(np.isinf(x2data)), np.where(np.isnan(ydataerror)), np.where(np.isinf(ydataerror))])
	print(indicestoignore)
	datatousenew = np.delete(datatouse, (indicestoignore), axis=0)

	ydata = datatousenew[:,0]
	ydataerror = datatousenew[:,9]
	x1data = datatousenew[:,y[0]]
	x2data = datatousenew[:,y[1]]
	x3data = datatousenew[:,y[2]]
	x4data = datatousenew[:,y[3]]
	x1dataerror = datatousenew[:,float(y[0])+9.0]
	x2dataerror = datatousenew[:,float(y[1])+9.0]
	x3dataerror = datatousenew[:,float(y[2])+9.0]
	x4dataerror = datatousenew[:,float(y[3])+9.0]
	
	starting_guesses = np.random.rand(nwalkers, ndim)
	pool = Pool(cores)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_four, args=[x1data, x2data, x3data, x4data, ydata, ydataerror, x1dataerror, x2dataerror, x3dataerror, x4dataerror, len(x1data)], pool=pool)
	sampler.run_mcmc(starting_guesses, nsteps)
	sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
	sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
	
	logposteriors = sampler.lnprobability[:, nburn:].reshape(-1)
	answers = np.mean(sample[np.where(logposteriors== np.max(logposteriors))],0).reshape(ndim)
	theta3 = answers
	
	figure = triangle.corner(sample, labels=[ headings[int(y[0])-1], headings[int(y[1])-1], headings[int(y[2])-1], headings[int(y[3])-1], 'Intercept'], truths=theta3)
	figure.savefig("triangle_four_dim_test_"+str(headings[int(y[0])-1])+"_"+str(headings[int(y[1])-1])+"_"+str(headings[int(y[2])-1])+"_"+str(headings[int(y[3])-1])+".png", format='png')

	print(theta3)
	print(chi_squared_four(theta3, x1data, x2data, x3data, x4data, ydata, ydataerror))
	print(pearsons_chi_squared_four(theta3, x1data, x2data, x3data, x4data, ydata))
	print(log_likelihood_simple_four(theta3, x1data, x2data, x3data, x4data, ydata, ydataerror,  x1dataerror, x2dataerror, x3dataerror, x4dataerror, len(x1data)))
	print(len(x1data))
	
	f = open('herschel_stats_fourdim_data.txt', 'a')
	f.write(str(theta3[0]))
	f.write(str("    "))
	f.write(str(theta3[1]))
	f.write(str("    "))
	f.write(str(theta3[2]))
	f.write(str("    "))
	f.write(str(theta3[3]))
	f.write(str("    "))
	f.write(str(theta3[4]))
	f.write(str("    "))
	f.write(str(chi_squared_four(theta3, x1data, x2data, x3data, x4data, ydata, ydataerror)))
	f.write(str("    "))
	f.write(str(pearsons_chi_squared_four(theta3, x1data, x2data, x3data, x4data, ydata)))
	f.write(str("    "))
	f.write(str(log_likelihood_simple_four(theta3, x1data, x2data, x3data, x4data, ydata, ydataerror,  x1dataerror, x2dataerror, x3dataerror, x4dataerror, len(x1data)))) #-(len(set(indicestoignore[0]))/2.0)*(1.837877)-np.sum(np.log(datatouse[list(set(indicestoignore[0])),10]))))
	f.write(str("    "))
	f.write(str(len(x1data)))
	f.write(str("    "))
	f.write(str(len(indicestoignore[0])))
	f.write(str("    "))
	f.write(str(len(set(indicestoignore[0]))))
	f.write("\n")
	f.close()

	sampler.pool.terminate()
	
	plt.clf()
	plt.close()

	del theta3
	del ydata
	del ydataerror
	del x1data
	del x2data
	del x3data
	del x4data
	del sample
	del sampler
	del figure
	


###############################Now to do the Bayesian fit for five parameters#################################
np.random.seed(0)
cores = 8
nparameters = 5
ndim = nparameters + 1				# this should be the number of parameters + 1.
nwalkers = 200
nburn = 500
nsteps = 16000
print('Number of free parameters in our model =', ndim)
datatouse = np.loadtxt("Python_herschel_analysis.dat")
f = open('herschel_stats_fivedim_data.txt', 'w')
f.close()
	

for i in range(len(list(itertools.combinations('12345678', 5)))):
	x = list(itertools.combinations('12345678', 5))				#three parameter hence why repeat equals 1
	y = list(x[i])
	print(y[0])	
	print(y[1])													#number of numbers in the combinatorix - 1 	for the first etc										
	print(y[2])
	print(y[3])
	print(y[4])
	
	ydata = datatouse[:,0]
	ydataerror = datatouse[:,9]
	x1data = datatouse[:,y[0]]
	x2data = datatouse[:,y[1]]
	x3data = datatouse[:,y[2]]
	x4data = datatouse[:,y[3]]
	x5data = datatouse[:,y[4]]
	x1dataerror = datatouse[:,float(y[0])+9.0]
	x2dataerror = datatouse[:,float(y[1])+9.0]
	x3dataerror = datatouse[:,float(y[2])+9.0]
	x4dataerror = datatouse[:,float(y[3])+9.0]
	x5dataerror = datatouse[:,float(y[4])+9.0]
	indicestoignore = np.hstack([np.where(np.isnan(ydata)), np.where(np.isinf(ydata)),   np.where(np.isnan(x5data)), np.where(np.isinf(x5data)),  np.where(np.isnan(x4data)), np.where(np.isinf(x4data)),  np.where(np.isnan(x3data)), np.where(np.isinf(x3data)),  np.where(np.isnan(x1data)), np.where(np.isinf(x1data)), np.where(np.isnan(x2data)), np.where(np.isinf(x2data)),  np.where(np.isnan(ydataerror)), np.where(np.isinf(ydataerror))])
	datatousenew = np.delete(datatouse, (indicestoignore), axis=0)

	ydata = datatousenew[:,0]
	ydataerror = datatousenew[:,9]
	x1data = datatousenew[:,y[0]]
	x2data = datatousenew[:,y[1]]
	x3data = datatousenew[:,y[2]]
	x4data = datatousenew[:,y[3]]
	x5data = datatousenew[:,y[4]]
	x1dataerror = datatousenew[:,float(y[0])+9.0]
	x2dataerror = datatousenew[:,float(y[1])+9.0]
	x3dataerror = datatousenew[:,float(y[2])+9.0]
	x4dataerror = datatousenew[:,float(y[3])+9.0]
	x5dataerror = datatousenew[:,float(y[4])+9.0]
	
	starting_guesses = np.random.rand(nwalkers, ndim)
	pool = Pool(cores)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_five, args=[x1data, x2data, x3data, x4data, x5data, ydata, ydataerror, x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, len(x1data)], pool=pool)
	sampler.run_mcmc(starting_guesses, nsteps)
	sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
	sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
	
	logposteriors = sampler.lnprobability[:, nburn:].reshape(-1)
	answers = np.mean(sample[np.where(logposteriors== np.max(logposteriors))],0).reshape(ndim)
	theta3 = answers
	
	figure = triangle.corner(sample, labels=[ headings[int(y[0])-1], headings[int(y[1])-1], headings[int(y[2])-1], headings[int(y[3])-1], headings[int(y[4])-1], 'Intercept'], truths=theta3)
	figure.savefig("triangle_five_dim_test_"+str(headings[int(y[0])-1])+"_"+str(headings[int(y[1])-1])+"_"+str(headings[int(y[2])-1])+"_"+str(headings[int(y[3])-1])+"_"+str(headings[int(y[4])-1])+".png", format='png')

	print(theta3)
	print(chi_squared_five(theta3, x1data, x2data, x3data, x4data, x5data, ydata, ydataerror))
	print(pearsons_chi_squared_five(theta3, x1data, x2data, x3data, x4data, x5data, ydata))
	print(log_likelihood_simple_five(theta3, x1data, x2data, x3data, x4data, x5data, ydata, ydataerror, x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, len(x1data)))
	print(len(x1data))
	
	f = open('herschel_stats_fivedim_data.txt', 'a')	
	f.write(str(theta3[0]))
	f.write(str("    "))
	f.write(str(theta3[1]))
	f.write(str("    "))
	f.write(str(theta3[2]))
	f.write(str("    "))
	f.write(str(theta3[3]))
	f.write(str("    "))
	f.write(str(theta3[4]))
	f.write(str("    "))
	f.write(str(theta3[5]))
	f.write(str("    "))
	f.write(str(chi_squared_five(theta3, x1data, x2data, x3data, x4data, x5data, ydata, ydataerror)))
	f.write(str("    "))
	f.write(str(pearsons_chi_squared_five(theta3, x1data, x2data, x3data, x4data, x5data, ydata)))
	f.write(str("    "))
	f.write(str(log_likelihood_simple_five(theta3, x1data, x2data, x3data, x4data, x5data, ydata, ydataerror, x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, len(x1data))))  #-(len(set(indicestoignore[0]))/2.0)*(1.837877)-np.sum(np.log(datatouse[list(set(indicestoignore[0])),10]))))
	f.write(str("    "))
	f.write(str(len(x1data)))
	f.write(str("    "))
	f.write(str(len(indicestoignore[0])))
	f.write(str("    "))
	f.write(str(len(set(indicestoignore[0]))))
	f.write("\n")
	f.close()

	sampler.pool.terminate()
	
	plt.clf()
	plt.close()

	del theta3
	del ydata
	del ydataerror
	del x1data
	del x2data
	del x3data
	del x4data
	del x5data
	del sample
	del sampler
	del figure
	
	
	
###############################Now to do the Bayesian fit for six parameters#################################
np.random.seed(0)
cores = 8
nparameters = 6
ndim = nparameters + 1				# this should be the number of parameters + 1.
nwalkers = 200
nburn = 500
nsteps = 16000
print('Number of free parameters in our model =', ndim)
datatouse = np.loadtxt("Python_herschel_analysis.dat")

f = open('herschel_stats_sixdim_data.txt', 'w')
f.close()

for i in range(len(list(itertools.combinations('12345678', 6)))):
	x = list(itertools.combinations('12345678', 6))				#three parameter hence why repeat equals 1
	y = list(x[i])
	print(y[0])	
	print(y[1])													#number of numbers in the combinatorix - 1 	for the first etc										
	print(y[2])
	print(y[3])
	print(y[4])
	print(y[5])
	
	ydata = datatouse[:,0]
	ydataerror = datatouse[:,9]
	x1data = datatouse[:,y[0]]
	x2data = datatouse[:,y[1]]
	x3data = datatouse[:,y[2]]
	x4data = datatouse[:,y[3]]
	x5data = datatouse[:,y[4]]
	x6data = datatouse[:,y[5]]
	x1dataerror = datatouse[:,float(y[0])+9.0]
	x2dataerror = datatouse[:,float(y[1])+9.0]
	x3dataerror = datatouse[:,float(y[2])+9.0]
	x4dataerror = datatouse[:,float(y[3])+9.0]
	x5dataerror = datatouse[:,float(y[4])+9.0]
	x6dataerror = datatouse[:,float(y[5])+9.0]
	indicestoignore = np.hstack([np.where(np.isnan(ydata)), np.where(np.isinf(ydata)), np.where(np.isnan(x6data)), np.where(np.isinf(x6data)), np.where(np.isnan(x5data)), np.where(np.isinf(x5data)), np.where(np.isnan(x4data)), np.where(np.isinf(x4data)), np.where(np.isnan(x3data)), np.where(np.isinf(x3data)),  np.where(np.isnan(x1data)), np.where(np.isinf(x1data)),  np.where(np.isnan(x2data)), np.where(np.isinf(x2data)), np.where(np.isnan(ydataerror)), np.where(np.isinf(ydataerror))])
	datatousenew = np.delete(datatouse, (indicestoignore), axis=0)

	ydata = datatousenew[:,0]
	ydataerror = datatousenew[:,9]
	x1data = datatousenew[:,y[0]]
	x2data = datatousenew[:,y[1]]
	x3data = datatousenew[:,y[2]]
	x4data = datatousenew[:,y[3]]
	x5data = datatousenew[:,y[4]]
	x6data = datatousenew[:,y[5]]
	x1dataerror = datatousenew[:,float(y[0])+9.0]
	x2dataerror = datatousenew[:,float(y[1])+9.0]
	x3dataerror = datatousenew[:,float(y[2])+9.0]
	x4dataerror = datatousenew[:,float(y[3])+9.0]
	x5dataerror = datatousenew[:,float(y[4])+9.0]
	x6dataerror = datatousenew[:,float(y[5])+9.0]
	
	starting_guesses = np.random.rand(nwalkers, ndim)
	pool = Pool(cores)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_six, args=[x1data, x2data, x3data, x4data, x5data, x6data, ydata, ydataerror, x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, x6dataerror, len(x1data)], pool=pool)
	sampler.run_mcmc(starting_guesses, nsteps)
	sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
	sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
	
	logposteriors = sampler.lnprobability[:, nburn:].reshape(-1)
	answers = np.mean(sample[np.where(logposteriors== np.max(logposteriors))],0).reshape(ndim)
	theta3 = answers
	
	
	figure = triangle.corner(sample, labels=[headings[int(y[0])-1], headings[int(y[1])-1], headings[int(y[2])-1], headings[int(y[3])-1], headings[int(y[4])-1], headings[int(y[5])-1], 'Intercept'], truths=theta3)
	figure.savefig("triangle_six_dim_test_"+str(headings[int(y[0])-1])+"_"+str(headings[int(y[1])-1])+"_"+str(headings[int(y[2])-1])+"_"+str(headings[int(y[3])-1])+"_"+str(headings[int(y[4])-1])+"_"+str(headings[int(y[5])-1])+".png", format='png')

	print(theta3)
	print(chi_squared_six(theta3, x1data, x2data, x3data, x4data, x5data, x6data, ydata, ydataerror))
	print(pearsons_chi_squared_six(theta3, x1data, x2data, x3data, x4data, x5data, x6data, ydata))
	print(log_likelihood_simple_six(theta3, x1data, x2data, x3data, x4data, x5data, x6data, ydata, ydataerror,  x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, x6dataerror, len(x1data)))
	print(len(x1data))
	
	f = open('herschel_stats_sixdim_data.txt', 'a')
	f.write(str(theta3[0]))
	f.write(str("    "))
	f.write(str(theta3[1]))
	f.write(str("    "))
	f.write(str(theta3[2]))
	f.write(str("    "))
	f.write(str(theta3[3]))
	f.write(str("    "))
	f.write(str(theta3[4]))
	f.write(str("    "))
	f.write(str(theta3[5]))
	f.write(str("    "))
	f.write(str(theta3[6]))
	f.write(str("    "))
	f.write(str(chi_squared_six(theta3, x1data, x2data, x3data, x4data, x5data, x6data, ydata, ydataerror)))
	f.write(str("    "))
	f.write(str(pearsons_chi_squared_six(theta3, x1data, x2data, x3data, x4data, x5data, x6data, ydata)))
	f.write(str("    "))
	f.write(str(log_likelihood_simple_six(theta3, x1data, x2data, x3data, x4data, x5data, x6data, ydata, ydataerror,  x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, x6dataerror, len(x1data))))  #-(len(set(indicestoignore[0]))/2.0)*(1.837877)-np.sum(np.log(datatouse[list(set(indicestoignore[0])),10]))))
	f.write(str("    "))
	f.write(str(len(x1data)))
	f.write(str("    "))
	f.write(str(len(indicestoignore[0])))
	f.write(str("    "))
	f.write(str(len(set(indicestoignore[0]))))
	f.write("\n")
	f.close()

	sampler.pool.terminate()
	
	plt.clf()
	plt.close()

	del theta3
	del ydata
	del ydataerror
	del x1data
	del x2data
	del x3data
	del x4data
	del x5data
	del x6data
	del sample
	del sampler
	del figure


###############################Now to do the Bayesian fit for seven parameters#################################
np.random.seed(0)
cores = 8
nparameters = 7
ndim = nparameters + 1				# this should be the number of parameters + 1.
nwalkers = 200
nburn = 500
nsteps = 16000
print('Number of free parameters in our model =', ndim)
datatouse = np.loadtxt("Python_herschel_analysis.dat")
	

f = open('herschel_stats_sevendim_data.txt', 'w')
f.close()



for i in range(len(list(itertools.combinations('12345678', 7)))):
	x = list(itertools.combinations('12345678', 7))				#three parameter hence why repeat equals 1
	y = list(x[i])
	print(y[0])	
	print(y[1])													#number of numbers in the combinatorix - 1 	for the first etc										
	print(y[2])
	print(y[3])
	print(y[4])
	print(y[5])
	print(y[6])
	
	ydata = datatouse[:,0]
	ydataerror = datatouse[:,9]
	x1data = datatouse[:,y[0]]
	x2data = datatouse[:,y[1]]
	x3data = datatouse[:,y[2]]
	x4data = datatouse[:,y[3]]
	x5data = datatouse[:,y[4]]
	x6data = datatouse[:,y[5]]
	x7data = datatouse[:,y[6]]
	x1dataerror = datatouse[:,float(y[0])+9.0]
	x2dataerror = datatouse[:,float(y[1])+9.0]
	x3dataerror = datatouse[:,float(y[2])+9.0]
	x4dataerror = datatouse[:,float(y[3])+9.0]
	x5dataerror = datatouse[:,float(y[4])+9.0]
	x6dataerror = datatouse[:,float(y[5])+9.0]
	x7dataerror = datatouse[:,float(y[6])+9.0]
	indicestoignore = np.hstack([np.where(np.isnan(ydata)), np.where(np.isinf(ydata)),np.where(np.isnan(x7data)), np.where(np.isinf(x7data)), np.where(np.isnan(x6data)), np.where(np.isinf(x6data)),np.where(np.isnan(x5data)), np.where(np.isinf(x5data)), np.where(np.isnan(x4data)), np.where(np.isinf(x4data)),  np.where(np.isnan(x3data)), np.where(np.isinf(x3data)), np.where(np.isnan(x1data)), np.where(np.isinf(x1data)), np.where(np.isnan(x2data)), np.where(np.isinf(x2data)),  np.where(np.isnan(ydataerror)), np.where(np.isinf(ydataerror))])
	datatousenew = np.delete(datatouse, (indicestoignore), axis=0)

	ydata = datatousenew[:,0]
	ydataerror = datatousenew[:,9]
	x1data = datatousenew[:,y[0]]
	x2data = datatousenew[:,y[1]]
	x3data = datatousenew[:,y[2]]
	x4data = datatousenew[:,y[3]]
	x5data = datatousenew[:,y[4]]
	x6data = datatousenew[:,y[5]]
	x7data = datatousenew[:,y[6]]

	x1dataerror = datatousenew[:,float(y[0])+9.0]
	x2dataerror = datatousenew[:,float(y[1])+9.0]
	x3dataerror = datatousenew[:,float(y[2])+9.0]
	x4dataerror = datatousenew[:,float(y[3])+9.0]
	x5dataerror = datatousenew[:,float(y[4])+9.0]
	x6dataerror = datatousenew[:,float(y[5])+9.0]
	x7dataerror = datatousenew[:,float(y[6])+9.0]
	
	starting_guesses = np.random.rand(nwalkers, ndim)
	pool = Pool(cores)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_seven, args=[x1data, x2data, x3data, x4data, x5data, x6data, x7data, ydata, ydataerror, x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, x6dataerror, x7dataerror, len(x1data)], pool=pool)
	sampler.run_mcmc(starting_guesses, nsteps)
	sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
	sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
	
	logposteriors = sampler.lnprobability[:, nburn:].reshape(-1)
	answers = np.mean(sample[np.where(logposteriors== np.max(logposteriors))],0).reshape(ndim)
	theta3 = answers
	
	figure = triangle.corner(sample, labels=[headings[int(y[0])-1], headings[int(y[1])-1], headings[int(y[2])-1], headings[int(y[3])-1], headings[int(y[4])-1], headings[int(y[5])-1],  headings[int(y[6])-1], 'Intercept'], truths=theta3)
	figure.savefig("triangle_seven_dim_test_"+str(headings[int(y[0])-1])+"_"+str(headings[int(y[1])-1])+"_"+str(headings[int(y[2])-1])+"_"+str(headings[int(y[3])-1])+"_"+str(headings[int(y[4])-1])+"_"+str(headings[int(y[5])-1])+"_"+str(headings[int(y[6])-1])+".png", format='png')
	theta3 = np.mean(sample, 0)

	print(chi_squared_seven(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data, ydata, ydataerror))
	print(pearsons_chi_squared_seven(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data, ydata))
	print(log_likelihood_simple_seven(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data, ydata, ydataerror,  x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, x6dataerror, x7dataerror, len(x1data)))
	print(len(x1data))
	
	f = open('herschel_stats_sevendim_data.txt', 'a')
	f.write(str(theta3[0]))
	f.write(str("    "))
	f.write(str(theta3[1]))
	f.write(str("    "))
	f.write(str(theta3[2]))
	f.write(str("    "))
	f.write(str(theta3[3]))
	f.write(str("    "))
	f.write(str(theta3[4]))
	f.write(str("    "))
	f.write(str(theta3[5]))
	f.write(str("    "))
	f.write(str(theta3[6]))
	f.write(str("    "))
	f.write(str(theta3[7]))
	f.write(str("    "))
	f.write(str(chi_squared_seven(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data, ydata, ydataerror)))
	f.write(str("    "))
	f.write(str(pearsons_chi_squared_seven(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data, ydata)))
	f.write(str("    "))
	f.write(str(log_likelihood_simple_seven(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data, ydata, ydataerror,  x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, x6dataerror, x7dataerror, len(x1data))))  #-(len(set(indicestoignore[0]))/2.0)*(1.837877)-np.sum(np.log(datatouse[list(set(indicestoignore[0])),10]))))
	f.write(str("    "))
	f.write(str(len(x1data)))
	f.write(str("    "))
	f.write(str(len(indicestoignore[0])))
	f.write(str("    "))
	f.write(str(len(set(indicestoignore[0]))))
	f.write("\n")
	f.close()

	plt.clf()
	plt.close()

	sampler.pool.terminate()
	
	del theta3
	del ydata
	del ydataerror
	del x1data
	del x2data
	del x3data
	del x4data
	del x5data
	del x6data
	del x7data
	del sample
	del sampler
	del figure

###############################Now to do the Bayesian fit for eight parameters#################################
np.random.seed(0)
cores = 8
nparameters = 8
ndim = nparameters + 1				# this should be the number of parameters + 1.
nwalkers = 200
nburn = 500
nsteps = 16000
print('Number of free parameters in our model =', ndim)
datatouse = np.loadtxt("Python_herschel_analysis.dat")


f = open('herschel_stats_eightdim_data.txt', 'w')
f.close()


for i in range(len(list(itertools.combinations('12345678', 8)))):
	x = list(itertools.combinations('12345678', 8))				#three parameter hence why repeat equals 1
	y = list(x[i])
	print(y[0])	
	print(y[1])													#number of numbers in the combinatorix - 1 	for the first etc										
	print(y[2])
	print(y[3])
	print(y[4])
	print(y[5])
	print(y[6])
	print(y[7])
	
	ydata = datatouse[:,0]
	ydataerror = datatouse[:,9]
	x1data = datatouse[:,y[0]]
	x2data = datatouse[:,y[1]]
	x3data = datatouse[:,y[2]]
	x4data = datatouse[:,y[3]]
	x5data = datatouse[:,y[4]]
	x6data = datatouse[:,y[5]]
	x7data = datatouse[:,y[6]]
	x8data = datatouse[:,y[7]]
	x1dataerror = datatouse[:,float(y[0])+9.0]
	x2dataerror = datatouse[:,float(y[1])+9.0]
	x3dataerror = datatouse[:,float(y[2])+9.0]
	x4dataerror = datatouse[:,float(y[3])+9.0]
	x5dataerror = datatouse[:,float(y[4])+9.0]
	x6dataerror = datatouse[:,float(y[5])+9.0]
	x7dataerror = datatouse[:,float(y[6])+9.0]
	x8dataerror = datatouse[:,float(y[7])+9.0]
	indicestoignore = np.hstack([np.where(np.isnan(ydata)), np.where(np.isinf(ydata)),  np.where(np.isnan(x8data)), np.where(np.isinf(x8data)),  np.where(np.isnan(x7data)), np.where(np.isinf(x7data)),  np.where(np.isnan(x6data)), np.where(np.isinf(x6data)), np.where(np.isnan(x5data)), np.where(np.isinf(x5data)), np.where(np.isnan(x4data)), np.where(np.isinf(x4data)), np.where(np.isnan(x3data)), np.where(np.isinf(x3data)),np.where(np.isnan(x1data)), np.where(np.isinf(x1data)),  np.where(np.isnan(x2data)), np.where(np.isinf(x2data)),  np.where(np.isnan(ydataerror)), np.where(np.isinf(ydataerror))])
	datatousenew = np.delete(datatouse, (indicestoignore), axis=0)

	ydata = datatousenew[:,0]
	ydataerror = datatousenew[:,9]
	x1data = datatousenew[:,y[0]]
	x2data = datatousenew[:,y[1]]
	x3data = datatousenew[:,y[2]]
	x4data = datatousenew[:,y[3]]
	x5data = datatousenew[:,y[4]]
	x6data = datatousenew[:,y[5]]
	x7data = datatousenew[:,y[6]]
	x8data = datatousenew[:,y[7]]

	x1dataerror = datatousenew[:,float(y[0])+9.0]
	x2dataerror = datatousenew[:,float(y[1])+9.0]
	x3dataerror = datatousenew[:,float(y[2])+9.0]
	x4dataerror = datatousenew[:,float(y[3])+9.0]
	x5dataerror = datatousenew[:,float(y[4])+9.0]
	x6dataerror = datatousenew[:,float(y[5])+9.0]
	x7dataerror = datatousenew[:,float(y[6])+9.0]
	x8dataerror = datatousenew[:,float(y[7])+9.0]
	
	starting_guesses = np.random.rand(nwalkers, ndim)
	pool = Pool(cores)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_eight, args=[x1data, x2data, x3data, x4data, x5data, x6data, x7data, x8data, ydata, ydataerror, x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, x6dataerror, x7dataerror, x8dataerror, len(x1data)], pool=pool)
	sampler.run_mcmc(starting_guesses, nsteps)
	sample = sampler.chain  # shape = (nwalkers, nsteps, ndim)
	sample = sampler.chain[:, nburn:, :].reshape(-1, ndim)
	
	logposteriors = sampler.lnprobability[:, nburn:].reshape(-1)
	answers = np.mean(sample[np.where(logposteriors== np.max(logposteriors))],0).reshape(ndim)
	theta3 = answers
	
	figure = triangle.corner(sample, labels=[headings[int(y[0])-1], headings[int(y[1])-1], headings[int(y[2])-1], headings[int(y[3])-1], headings[int(y[4])-1], headings[int(y[5])-1],  headings[int(y[6])-1], headings[int(y[7])-1], 'Intercept'], truths=theta3)
	figure.savefig("triangle_eight_dim_test_"+str(headings[int(y[0])-1])+"_"+str(headings[int(y[1])-1])+"_"+str(headings[int(y[2])-1])+"_"+str(headings[int(y[3])-1])+"_"+str(headings[int(y[4])-1])+"_"+str(headings[int(y[5])-1])+"_"+str(headings[int(y[6])-1])+"_"+str(headings[int(y[7])-1])+".png", format='png')

	print(theta3)
	print(chi_squared_eight(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data,  x8data, ydata, ydataerror))
	print(pearsons_chi_squared_eight(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data,  x8data, ydata))
	print(log_likelihood_simple_eight(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data, ydata,  x8data, ydataerror,  x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, x6dataerror, x7dataerror, x8dataerror, len(x1data)))			#there is an error on this line
	print(len(x1data))
	
	f = open('herschel_stats_eightdim_data.txt', 'a')
	f.write(str(theta3[0]))
	f.write(str("    "))
	f.write(str(theta3[1]))
	f.write(str("    "))
	f.write(str(theta3[2]))
	f.write(str("    "))
	f.write(str(theta3[3]))
	f.write(str("    "))
	f.write(str(theta3[4]))
	f.write(str("    "))
	f.write(str(theta3[5]))
	f.write(str("    "))
	f.write(str(theta3[6]))
	f.write(str("    "))
	f.write(str(theta3[7]))
	f.write(str("    "))
	f.write(str(theta3[8]))
	f.write(str("    "))
	f.write(str(chi_squared_eight(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data, x8data, ydata, ydataerror)))
	f.write(str("    "))
	f.write(str(pearsons_chi_squared_eight(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data, x8data, ydata)))
	f.write(str("    "))
	f.write(str(log_likelihood_simple_eight(theta3, x1data, x2data, x3data, x4data, x5data, x6data, x7data, x8data, ydata, ydataerror,  x1dataerror, x2dataerror, x3dataerror, x4dataerror, x5dataerror, x6dataerror, x7dataerror, x8dataerror, len(x1data))))  #-(len(set(indicestoignore[0]))/2.0)*(1.837877)-np.sum(np.log(datatouse[list(set(indicestoignore[0])),10]))))
	f.write(str("    "))
	f.write(str(len(x1data)))
	f.write(str("    "))
	f.write(str(len(indicestoignore[0])))
	f.write(str("    "))
	f.write(str(len(set(indicestoignore[0]))))
	f.write("\n")
	f.close()

	sampler.pool.terminate()
	
	plt.clf()
	plt.close()

	del theta3
	del ydata
	del ydataerror
	del x1data
	del x2data
	del x3data
	del x4data
	del x5data
	del x6data
	del x7data
	del x8data
	del sample
	del sampler
	del figure


















