from turtle import tilt
import numpy as np
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#plots for intensityu at screen comparison, between tilted and not tilted filter holders in front of ligth crafter

position= ['center','center','center','left_up','left_up','left_up','rigth_down','rigth_down','rigth_down','center','center','center','left_up','left_up','left_up','rigth_down','rigth_down','rigth_down']
tilt_=[0,0,0,0,0,0,0,0,0,15,15,15,15,15,15,15,15,15]
value=[2.34,3.38,2.37,3.77,3.74,3.84,1.69,1.68,1.69,1.35,1.32,1.37,1.18,1.15,1.2,1.23,1.23,1.24]

data_dict={'position':['center','center','center','left_up','left_up','left_up','rigth_down','rigth_down','rigth_down','center','center','center','left_up','left_up','left_up','rigth_down','rigth_down','rigth_down'],
    'tilt':[0,0,0,0,0,0,0,0,0,15,15,15,15,15,15,15,15,15],'value':[2.34,3.38,2.37,3.77,3.74,3.84,1.69,1.68,1.69,1.35,1.32,1.37,1.18,1.15,1.2,1.23,1.23,1.24]}


dataframe_=pd.DataFrame.from_dict(data_dict)
#dataframe_15deg=pd.DataFrame.from_dict(tilt_15deg)

subset1=dataframe_.loc[dataframe_['tilt']==0]
subset2=dataframe_.loc[dataframe_['tilt']==15]

plt.figure()
sns.barplot(data=subset1,x='position',y='value',facecolor='steelblue')
plt.title(' 0deg tilt')
plt.ylabel('intensity (uW/cm2)')
plt.close()
plt.figure()
sns.barplot(data=subset2,x='position',y='value',facecolor='steelblue')
plt.title(' 15deg tilt')
plt.ylabel('intensity (uW/cm2)')


### plots for the calibration of the new pvc140 screen material

path=r'C:\Users\vargasju\PhD\troubleshooting\pvc140_calibration_ultima.csv'

cal_df=pd.read_csv(path)

subset1=cal_df.loc[cal_df['screen_type']=='pvc-140b']
subset2=cal_df.loc[cal_df['screen_type']=='pvc']


plt.close('all')
plt.figure()
sns.lineplot(data=cal_df,x='current',y='value at center (uW)',hue='screen_type',err_style='bars')
sns.scatterplot(data=cal_df,x='current',y='value at center (uW)',hue='screen_type')
