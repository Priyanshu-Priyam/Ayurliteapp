import joblib
import numpy as np
import pandas as pd
from numpy import loadtxt
from scipy.signal import savgol_filter
import os
from sklearn.preprocessing import normalize
from scipy.signal import argrelmax, argrelmin




def blood_pressure(X):

  

    model = joblib.load('bp_prediction_api_test.pkl') 


    single_waveform1= X
    single_waveform1=single_waveform1.astype('float64')
    single_waveform=single_waveform1.flatten()
    sample_rate=20



    maxima_index = argrelmax(np.array(single_waveform))[0]
    minima_index = argrelmin(np.array(single_waveform))[0]
    derivative_1 = np.diff(single_waveform, n=1) * float(sample_rate)
    derivative_1_maxima_index = argrelmax(np.array(derivative_1))[0]
    derivative_1_minima_index = argrelmin(np.array(derivative_1))[0]
    derivative_2 = np.diff(single_waveform, n=2) * float(sample_rate)
    derivative_2_maxima_index = argrelmax(np.array(derivative_2))[0]
    derivative_2_minima_index = argrelmin(np.array(derivative_2))[0]


    x_train = np.zeros((1,38))

    j=0 
    x = single_waveform[maxima_index[0]]
    x_train[j,0]=x
    # y
    y = single_waveform[maxima_index[1]]
    x_train[j,1]=y
    # z
    z = single_waveform[minima_index[0]]
    x_train[j,2]=z
    # t_pi
    t_pi = float(len(single_waveform)) / float(sample_rate)
    x_train[j,3]=t_pi
    # y/x
    x_train[j,4]=(y / x)
    # (x-y)/x
    x_train[j,5]=((x - y) / x)
    # z/x
    x_train[j,6]=(z / x)
    # (y-z)/x
    x_train[j,7]=((y - z) / x)
    # t_1
    t_1 = float(maxima_index[0] + 1) / float(sample_rate)
    x_train[j,8]=(t_1)
    # t_2
    t_2 = float(minima_index[0] + 1) / float(sample_rate)
    x_train[j,9]=(t_2)
    # t_3
    t_3 = float(maxima_index[1] + 1) / float(sample_rate)
    x_train[j,10]=(t_3)
    # delta_t
    delta_t = t_3 - t_2
    x_train[j,11]=(delta_t)

    # A_2/A_1
    x_train[j,12]=(sum(single_waveform[:maxima_index[0]]) / sum(single_waveform[maxima_index[0]:]))
    # t_1/x
    x_train[j,13]=(t_1 / x)
    # y/(t_pi-t_3)
    x_train[j,14]=(y / (t_pi - t_3))
    # t_1/t_pi
    x_train[j,15]=(t_1 / t_pi)
    # t_2/t_pi
    x_train[j,16]=(t_2 / t_pi)
    # t_3/t_pi
    x_train[j,17]=(t_3 / t_pi)
    # delta_t/t_pi
    x_train[j,18]=(delta_t / t_pi)
    # t_a1
    t_a1 = float(derivative_1_maxima_index[0]) / float(sample_rate)
    x_train[j,19]=(t_a1)
    # t_b1
    t_b1 = float(derivative_1_minima_index[0]) / float(sample_rate)
    x_train[j,20]=(t_b1)
    # t_e1
    t_e1 = float(derivative_1_maxima_index[1]) / float(sample_rate)
    x_train[j,21]=(t_e1)
    # t_f1
    t_f1 = float(derivative_1_minima_index[1]) / float(sample_rate)
    x_train[j,22]=(t_f1)
    # b_2/a_2
    a_2 = derivative_2[derivative_2_maxima_index[0]]
    b_2 = derivative_2[derivative_2_minima_index[0]]
    x_train[j,23]=(b_2 / a_2)
    # e_2/a_2
    e_2 = derivative_2[derivative_2_maxima_index[1]]
    x_train[j,24]=(e_2 / a_2)
    # (b_2+e_2)/a_2
    x_train[j,25]=((b_2 + e_2) / a_2)
    # t_a2
    t_a2 = float(derivative_2_maxima_index[0]) / float(sample_rate)
    x_train[j,26]=(t_a2)
    # t_b2
    t_b2 = float(derivative_2_minima_index[0]) / float(sample_rate)
    x_train[j,27]=(t_b2)
    # t_a1/t_pi
    x_train[j,28]=(t_a1 / t_pi)
    # t_b1/t_pi
    x_train[j,29]=(t_b1 / t_pi)
    # t_e1/t_pi
    x_train[j,30]=(t_e1 / t_pi)
    # t_f1/t_pi
    x_train[j,31]=(t_f1 / t_pi)
    # t_a2/t_pi
    x_train[j,32]=(t_a2 / t_pi)
    # t_b2/t_pi
    x_train[j,33]=(t_b2 / t_pi)
    # (t_a1-t_a2)/t_pi
    x_train[j,34]=((t_a1 - t_a2) / t_pi)
    # (t_b1-t_b2)/t_pi
    x_train[j,35]=((t_b1 - t_b2) / t_pi)
    # (t_e1-t_2)/t_pi
    x_train[j,36]=((t_e1 - t_2) / t_pi)
    # (t_f1-t_3)/t_pi
    x_train[j,37]=((t_f1 - t_3) / t_pi)        
       

    prediction=model.predict(x_train)

    if prediction==0:
        con= "Your BP is in the range of Hypotension and You are Diabetic"
    elif prediction==1:
        con = "Your Bp is the range of Normal Blood Pressure"
    elif prediction==2:
        con =" Your Bp is the range of Elevated Blood Pressure"
    else:
        con = " Your Bp is the range of Hypertension are Non-Diabetic"
            
    return con





