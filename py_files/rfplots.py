#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

rfdata = pd.read_csv(r'C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\rfdata.csv', sep=",")
indeksi = pd.read_csv(r'C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\ftse.csv', sep=";", decimal = ",",index_col='Date', parse_dates=True, dayfirst=True)
rfdata["Name"] = pd.to_datetime(rfdata['Name'], format = '%Y-%m-%d')
rfdata["Indeksi"] = rfdata["Name"].dt.strftime("%d.%m.%y")
rfdata["Indeksi"] = pd.to_datetime(rfdata['Indeksi'], format = "%d.%m.%y")
rfdata["RF"] = pd.to_numeric(rfdata["RF"],errors = 'coerce')
rfdata["RF"] = rfdata["RF"].div(100).round(3)
rfdata = rfdata.dropna()

print(rfdata)

print(indeksi)

#indeksi["FTSE100"] = indeksi["FTSE100"].str.replace(',','.')
#indeksi["FTSE100"] = indeksi["FTSE100"].astype("float")
#indeksi["FTSE100"] = indeksi["FTSE100"].apply(lambda x: x.replace(',', '.')).astype('float')
indeksi["FTSE100"] = indeksi["FTSE100"].apply(pd.to_numeric)
#indeksi["Date"] = indeksi["Date"].apply(pd.to_datetime)

#pd.to_datetime(indeksi["Date"], format = "%d.%m.%y")


def plotting_rfrate(rfdata):

    #plotti = rfdata.plot("Indeksi", "RF")


    plt.figure(figsize=(10, 5))

    plt.plot(rfdata["Indeksi"],rfdata["RF"], color = "black")
    ax = plt.gca()
    plt.legend(['Risk-Free Rate'])
    plt.xlabel("Time")
    plt.ylabel("Risk-Free Rate")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval = 3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y"))
    plt.gcf().autofmt_xdate(rotation = 30)


    plt.show()

plotting_rfrate(rfdata)



def plotting_index(indeksi):


    plt.figure(figsize=(10, 5))
    plt.plot(indeksi.index,indeksi.FTSE100, color = "black")
    ax = plt.gca()
    plt.legend(['FTSE 100'])
    plt.xlabel("Time")
    plt.ylabel("index level")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval = 3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate(rotation = 30)
    plt.show()

plotting_index(indeksi)







# In[ ]:




