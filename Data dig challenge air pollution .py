# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:17:00 2019

@author: Ardian Grezda
"""

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


plt.style.use('ggplot')
df = pd.read_csv(r'KosovoAQ_raw (1).csv', 
                 index_col = 'date')
df.index = pd.to_datetime(df.index) 



ncol = 0
nrow = 5

fig, ax = plt.subplots(nrow + 1, 1, figsize=(12, 30), squeeze = False )
fig.autofmt_xdate()

nCurrentRow = 1
for param in df.parameter.unique():
    df1 = df[df['parameter'] == param]
    df1.head()
    for d in df1.location.unique():
        a = df1[df1['location'] == d]
        b = a.resample('D').max()
        ax[nCurrentRow - 1, 0].plot(b.index.values, b.value, label = d, linewidth=2)
        ax[nCurrentRow - 1, 0].legend(loc='upper right', title = param, frameon=True)
        ax[nCurrentRow - 1,0].set_ylabel('value ')
        ax[nCurrentRow - 1,0].set_title( param)
    nCurrentRow += 1

fig.autofmt_xdate()
fig.tight_layout()
plt.show()
fig.savefig("Values divided by stations and factors.jpg")

df3 = pd.DataFrame()
df3 = df[df.location == '']
df3.reset_index(inplace = True)
i = 0
for d in df.location.unique():
    left = df3
    right = df[df.location == d]
    right.reset_index(inplace = True)
    if i == 0:
        df3 = df[df.location == d][['parameter', 'location', 'value']]
        df3.rename(columns={'location':'LOC=%s'% df3.location[0], 'value':'VAL=%s'%df3.location[0]}, inplace=True)
        df3.reset_index(inplace=True)
    else:
        df3  = pd.merge (left =left, right=right, on =['date', 'parameter'], how='outer')
        df3.rename(columns={'location':'LOC=%s'% right.location[0], 'value':'VAL=%s'%right.location[0]}, inplace=True)
    i = i + 1

df1 = df[df.parameter == 'pm25']
df1[df1.location == 'Drenas'].plot(figsize=(10,9),
   title = 'Value for pm25 as measured by Drenas ')
plt.savefig('pm25 Drenas.jpg')

df1[df1.location == 'US Diplomatic Post: Pristina'].plot(figsize=(10,9),
   title = 'Value for pm25 as measured by US Ambasy (not removing values -999) ')
plt.savefig('pm25 US ambasy not removing -999.jpg')
    
df1[(df1.location == 'US Diplomatic Post: Pristina') & (df1.value > 0)].plot(figsize=(10,9), 
     title = 'Value for pm25 as measured by US Ambasy ')    
plt.savefig('pm25 US ambasy.jpg')

df2 = df1[(df1.location == 'US Diplomatic Post: Pristina') & (df1.value > 0)]

cons_pm25 = [0, 12, 35.5,55.5,150.5,250.5,800]


def label_pm25(x, cons_pm25):
    if x < cons_pm25[1]:
        return [0, "Good"]
    elif x < cons_pm25[2]:
        return [1, "Moderate"]
    elif x < cons_pm25[3]:
        return [2, "Unhealthy for Sensitive Groups"]
    elif x < cons_pm25[4]:
        return [3, "Unhealthy for Everyone"]
    elif x < cons_pm25[5]:
        return [4, "Very Unhealthy"]
    elif x < cons_pm25[6]:
        return [5, "Hazardous"]
    return [0, "Good"]


df2.resample('M').agg(['min','max','mean']).plot(figsize=(9,7), title = 'Minimum maximum and mean values by month')

def season(x):
    if x == 1 or x == 2 or x == 3 or x ==4 or x == 10 or x == 11 or x == 12:
        return "winter"
    else:
        return "summer"
    
# set season for summer/winter period    
df2['b'] = df2.index.strftime('%m').astype(int)
df2['season'] = df2.b.apply(season)
del df2['b']

df_summer = df2[df2.season == 'summer']
df_winter = df2[df2.season == 'winter']
df_summer_group = df_summer.groupby([df_summer.index.weekday, 
                                       df_summer.index.hour]).mean()
df_summer_group.rename(columns = {'value':'Summer'}, inplace=True)
df_winter_group = df_winter.groupby([df_winter.index.weekday, 
                                       df_winter.index.hour]).mean()
df_winter_group.rename(columns = {'value':'Winter'}, inplace = True)

df_agg = pd.concat([df_winter_group, df_summer_group],  axis = 1)

df_agg = df_agg.rename_axis(['weekday', 'hour'])
df_agg.reset_index(inplace=True)

cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#sorterIndex = dict(zip(range(len(cats)), cats))
#df_agg['weekday'] = df_agg['weekday'].map(sorterIndex)

df_agg.set_index(['weekday','hour'], inplace = True)


def plot_function(x, ax):
    ax = graph[x]
    ax.set_xlabel(x)
    return df_agg.xs(x).plot(kind='line',  ax=ax, title=cats[x])

n_subplots = len(df_agg.index.levels[0])
fig, axes = plt.subplots(nrows=1, ncols=n_subplots, sharey=True, figsize=(12, 8))  # width, height

graph = dict(zip(df_agg.index.levels[0], axes))
plots = list(map(lambda x: plot_function(x, graph[x]), graph))
#axes.tick_params(axis='both', which='both', length=0)
fig.subplots_adjust(wspace=0)

plt.legend()
plt.savefig('pm25 US ambasy by hour, week and season.jpg')
plt.show()

df4 = df2.resample('D').max()
df4['status'] = df4['value'].apply(lambda x: label_pm25(x, cons_pm25)[0])
df4 = df4['status']


DAYS = ['Sun.', 'Mon.', 'Tues.', 'Wed.', 'Thurs.', 'Fri.', 'Sat.']
MONTHS = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'June', 'July', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.']

def date_heatmap(series, start=None, end=None, mean=False, ax=None, **kwargs):
    '''Plot a calendar heatmap given a datetime series.

    Arguments:
        series (pd.Series):
            A series of numeric values with a datetime index. Values occurring
            on the same day are combined by sum.
        start (Any):
            The first day to be considered in the plot. The value can be
            anything accepted by :func:`pandas.to_datetime`. The default is the
            earliest date in the data.
        end (Any):
            The last day to be considered in the plot. The value can be
            anything accepted by :func:`pandas.to_datetime`. The default is the
            latest date in the data.
        mean (bool):
            Combine values occurring on the same day by mean instead of sum.
        ax (matplotlib.Axes or None):
            The axes on which to draw the heatmap. The default is the current
            axes in the :module:`~matplotlib.pyplot` API.
        **kwargs:
            Forwarded to :meth:`~matplotlib.Axes.pcolormesh` for drawing the
            heatmap.

    Returns:
        matplotlib.collections.Axes:
            The axes on which the heatmap was drawn. This is set as the current
            axes in the `~matplotlib.pyplot` API.
    '''
    # Combine values occurring on the same day.
    dates = series.index.floor('D')
    group = series.groupby(dates)
    series = group.mean() if mean else group.sum()

    # Parse start/end, defaulting to the min/max of the index.
    start = pd.to_datetime(start or series.index.min())
    end = pd.to_datetime(end or series.index.max())
    # We use [start, end) as a half-open interval below.
    end += np.timedelta64(1, 'D')

    # Get the previous/following Sunday to start/end.
    # Pandas and numpy day-of-week conventions are Monday=0 and Sunday=6.
    start_sun = start - np.timedelta64((start.dayofweek + 1) % 7, 'D')
    end_sun = end + np.timedelta64(7 - end.dayofweek - 1, 'D')

    # Create the heatmap and track ticks.
    num_weeks = (end_sun - start_sun).days // 7
    heatmap = np.zeros((7, num_weeks))
    ticks = {}  # week number -> month name
    for week in range(num_weeks):
        for day in range(7):
            date = start_sun + np.timedelta64(7 * week + day, 'D')
            if date.day == 1:
                ticks[week] = MONTHS[date.month - 1]
            if date.dayofyear == 1:
                ticks[week] += f'\n{date.year}'
            if start <= date < end:
                heatmap[day, week] = series.get(date, 0)
    # Get the coordinates, offset by 0.5 to align the ticks.
    y = np.arange(8) - 0.5
    x = np.arange(num_weeks + 1) - 0.5

    # Plot the heatmap. Prefer pcolormesh over imshow so that the figure can be
    # vectorized when saved to a compatible format. We must invert the axis for
    # pcolormesh, but not for imshow, so that it reads top-bottom, left-right.
    ax = ax or plt.gca()
    mesh = ax.pcolormesh(x, y, heatmap, **kwargs)
    ax.invert_yaxis()

    # Set the ticks.
    ax.set_xticks(list(ticks.keys()))
    ax.set_xticklabels(list(ticks.values()))
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels(DAYS)

    # Set the current image and axes in the pyplot API.
    plt.sca(ax)
    plt.sci(mesh)

    return ax

bar_levels = ['Good','Moderate','Unhealthy for Sensitive Groups','Unhealthy for Everyone','Very Unhealthy','','Hazardous']


figsize = plt.figaspect(7 / 56)
fig = plt.figure(figsize=figsize)
ax = date_heatmap(df4, start='2016-01-01', end='2016-12-31', edgecolor='black')
cbar = plt.colorbar(pad=0.02)
cbar.set_ticks(ticks=range(6))
cbar.set_ticklabels(bar_levels)
cmap = mpl.cm.get_cmap('Blues', 6)
plt.set_cmap(cmap)
plt.clim(-0.5, 4.5)
ax.set_aspect('equal')
plt.savefig('pm25 US ambasy shown in calendar for 2016.jpg')

figsize = plt.figaspect(7 / 56)
fig = plt.figure(figsize=figsize)
ax = date_heatmap(df4, start='2017-01-01', end='2017-12-31', edgecolor='black')
cbar = plt.colorbar(pad=0.02)
cbar.set_ticks(ticks=range(6))
cbar.set_ticklabels(bar_levels)
cmap = mpl.cm.get_cmap('Blues', 6)
plt.set_cmap(cmap)
plt.clim(-0.5, 4.5)
ax.set_aspect('equal')
plt.savefig('pm25 US ambasy shown in calendar for 2017.jpg')

figsize = plt.figaspect(7 / 56)
fig = plt.figure(figsize=figsize)
ax = date_heatmap(df4, start='2018-01-01', end='2018-12-31', edgecolor='black')
cbar = plt.colorbar(pad=0.02)
cbar.set_ticks(ticks=range(6))
cbar.set_ticklabels(bar_levels)
cmap = mpl.cm.get_cmap('Blues', 6)
plt.set_cmap(cmap)
plt.clim(-0.5, 4.5)
ax.set_aspect('equal')
plt.savefig('pm25 US ambasy shown in calendar for 2018.jpg')

figsize = plt.figaspect(7 / 56)
fig = plt.figure(figsize=figsize)
ax = date_heatmap(df4, start='2019-01-01', end='2019-12-31', edgecolor='black')
cbar = plt.colorbar(pad=0.02)
cbar.set_ticks(ticks=range(6))
cbar.set_ticklabels(bar_levels)
cmap = mpl.cm.get_cmap('Blues', 6)
plt.set_cmap(cmap)
plt.clim(-0.5, 4.5)
ax.set_aspect('equal')
plt.savefig('pm25 US ambasy shown in calendar for 2019.jpg')
