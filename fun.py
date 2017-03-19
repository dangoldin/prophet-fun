#! /usr/bin/env python

import matplotlib.pylab
import pandas as pd
import numpy as np
from fbprophet import Prophet

import pdb

def sleep_duration(row):
    sleep = row['Bed']
    wakeup = row['Wakeup']
    if sleep < wakeup:
        return wakeup - sleep
    return wakeup - (sleep - 12.0)

df = pd.read_csv('stats2016.csv')
df['hours_slept'] = df.apply(sleep_duration, axis=1)

# Force renames for prophet
df['y'] = df['hours_slept']
df['ds'] = df['Date']

print df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
print future.tail()

forecast = m.predict(future)
print forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
print forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# TODO: Figure out plotting
m.plot(forecast)
m.plot_components(forecast)
