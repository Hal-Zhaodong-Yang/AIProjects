# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 23:33:51 2021

@author: Hal Yang
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / np.power(2 * math.pi * sig **2, 1/2)

part23 = pd.read_csv("part23_data.csv", names = ['a1', 'a2', 'a3', 'a4', 'y'])

#print(part23)

i = 0
dataset1 = np.array(part23)[0:761].take([i,4],1)
dataset2 = np.array(part23)[762:].take([i,4],1)
dataset = np.array(part23)[:].take([i,4],1)

print(dataset[np.where(dataset[:, 0] < -2), 1])
dataset1 = dataset1[:,0]
dataset2 = dataset2[:,0]

mean1 = np.mean(dataset1)
std1 = np.std(dataset1)
mean2 = np.mean(dataset2)
std2 = np.std(dataset2)

print("mean1",mean1)
print("mean2",mean2)
print("middle",(mean1+mean2)/2)
print("weighted",(mean1*std1 + mean2 * std2) / (std1 + std2))

#print(mean1)
x_values = np.linspace(-8, 8, 120)




plt.figure()

plt.hist(dataset1,density = True, alpha = 0.8, color = "tab:orange")
plt.hist(dataset2,density = True, alpha = 0.8)
plt.plot(x_values, gaussian(x_values, mean1, std1), color = "tab:orange", linewidth = 3)
plt.plot(x_values, gaussian(x_values, mean2, std2), color = "tab:blue", linewidth = 3)

plt.figure()
plt.plot(x_values, gaussian(x_values, mean1, std1) - gaussian(x_values, mean2, std2))


diff = gaussian(x_values, mean1, std1) - gaussian(x_values, mean2, std2)
diff = np.absolute(diff)
print(x_values[np.argwhere(diff == np.min(diff))[0]])

x = np.arange(6).reshape(2, 3)
print(x)
#print(np.transpose(x))
#print(np.mean(x, axis = 0))
y = [0, 1, 2, 3, 4, 5]
y = np.array(y)
print(np.where(x[0] > 1))
print(np.where(y > 2, 1, 0))

#print(dataset[[0, 3, 34, 23, 235, 2, 1000, 5, 9], 1])



    