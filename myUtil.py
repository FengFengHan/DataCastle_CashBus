# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 21:37:54 2016

@author: HAN
"""
#import pandas as pd
# create dummy variable for regression

from sympy import  *
y1, y2, lam = symbols('y1 y2 lam')
f = y1 * log(1+ exp(-y2)) + (1-y1)*log(1 + exp(y2))
d1 = diff(f, y2)
d2 = diff(d1, y2)
ans = simplify(d2 /(d1 + lam) )
#(y1 + (-y1 + (y1 - 1)*exp(y2))*(exp(y2) + 1) - (y1 - 1)*exp(2*y2))/
#((exp(y2) + 1)*(-lam*(exp(y2) + 1) + y1 + (y1 - 1)*exp(y2)))