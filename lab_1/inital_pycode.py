# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:47:30 2025

@author: Anna Bernardi, Liam Kniberg

"""
import numpy as np 
import matplotlib.pyplot as plt


#The following code approximates the log_function using an itertative method as described in:
    
# Carlson, B. C. (1972). An algorithm for computing logarithms and arctangents. 
#    Mathematics of Computation, 26(118), 543–549. https://doi.org/10.1090/s0025-5718-1972-0307438-2


#Task 1: aprx the ln by n steps using the iterative method cited above------------------------------------------------------------------------------


def approx_ln(x,n):
    '''
    The function takes in an x value and returns the approximated ln(x) ≈ (x-1)/a_n 
    after n iterations, as described in https://doi.org/10.1090/s0025-5718-1972-0307438-2
    '''
    #initializing starting values
    a0 = (1 + x)/2
    g0 = np.sqrt(x)
    
    if x>0:
        for i in range(n+1):
            ai = (a0 + g0)/2
            gi= np.sqrt(ai * g0)
        
            a0 = ai
            g0 = gi
    else:
        raise Exception(f'{x} is not greater than x, the ln(x) is not defined')
        
    return (x-1)/a0


#Task 2: plotting ln and approx_ln------------------------------------------------------------------------------------------------------------------ 

x_asse = np.linspace(0.1, 50, 100) #starting at 0.1 because ln(0) is not defined
n = 7

#the approx_ln works with scalars so it needs to be vectorized to use it with an array
vec_approx_ln = np.vectorize(approx_ln)


fig, ax = plt.subplots()
plt.title('ln(x) and approx_ln for n iterations')

for i in range(2,n+1):
    ax.plot(x_asse, vec_approx_ln(x_asse, i), label=f'n = {i}')
 
    


ax.plot(x_asse,np.log(x_asse), 'k--', label='ln(x)') 



#zoomed in portion 
x1 = x_asse[90:]

zoom = ax.inset_axes([0.55,0.1,0.4,0.4]) #loc and size

for i in range(2,n+1):
    zoom.plot(x1,vec_approx_ln(x_asse, i)[90:])

zoom.plot(x1, np.log(x_asse)[90:], 'k--')
ax.indicate_inset_zoom(zoom)

plt.xlabel('x')
plt.ylabel('ln(x)') 
plt.legend(loc=2)  
plt.show()


#plotting the difference between the ln(x) and the aprx

plt.figure()
for i in range(2,n+1):
    plt.plot(x_asse, np.log(x_asse)-vec_approx_ln(x_asse, i), label=f'n = {i}')


plt.title('Differences between ln(x) and approx_ln for n iterations')
plt.xlabel('x')
plt.xlim(1)
plt.ylabel('ln(x) - aprx_ln(x)') 
plt.legend()
plt.show()  


#Task 3: plotting the absolute value for a specific x value------------------------------------------------------------------------------------------

x = 1.41
error = []
for i in range(2,n+1):
    error.append(np.abs(np.log(x)-approx_ln(x, i)))
    

plt.plot(np.linspace(2,n,6), error, label='ln(1.41) - aprx_ln(1.41)')
plt.title('Absolute value of the error vs number of interations')
plt.xlabel('n')
plt.ylabel('abs error')
plt.legend()
plt.show() 


#Task 4: faster aprx-------------------------------------------------------------------------------------------------------------------------------

def fast_approx_ln(x,n):
    '''
    The function is the accelerated Carlsson method to approximate the log.
    It takes a x value and returns the approximated ln(x) ≈ (x-1)/d_n_n 
    after n iterations, as described in https://doi.org/10.1090/s0025-5718-1972-0307438-2
    '''
    #initializing starting values
    a0 = (1 + x)/2
    g0 = np.sqrt(x)
    
    #and a list to store d[i][k]
    d = []   
    #it's a list of lists where:
    #   d[i][0] is the apprx we'd get from approx_ln after i iterations
    #   d[i][k] is the value at i-iteration and k-level                     
    
    for i in range(n+1):
        ai = (a0 + g0)/2
        gi= np.sqrt(ai * g0)
        
        a0 = ai
        g0 = gi
        
        #initial d[i][0] value
        d.append([ai])
        
        for k in range(1,i+1):
            dik = (d[i][k-1] - 4**(-k) * d[i-1][k-1]) / (1 - 4**(-k))
        
            d[i].append(dik)
        
    return (x-1)/d[n][n]

#Task 5: error plot for the faster aprx------------------------------------------------------------------------------------------------------------ 

x_asse = np.linspace(0.1, 20, 1000)  

vec_fast_approx_ln = np.vectorize(fast_approx_ln)


plt.figure()


for i in range(2,n):
    error = np.abs(np.log(x_asse) - vec_fast_approx_ln(x_asse, i))
    
    plt.scatter(x_asse, error, label=f'Iteration {i}', s=10)



plt.title('Error behavior of the accelerated Carlsson method for the log')
plt.xlabel('x')
plt.ylabel('abs error')
plt.yscale('log') #plotting the y axis in log scale to match the given plot
plt.ylim(1e-19,1e-5)
plt.xlim(0.0,20.0)
plt.legend()


plt.show()




