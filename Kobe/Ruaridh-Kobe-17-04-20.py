# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 17:51:34 2020

@author: bubba_000
"""

# Kobe

# https://www.kaggle.com/dixhom/data-analysis-for-beginners

import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
import random
import seaborn as sns
%matplotlib inline

random.seed(123)

##### Functions



#####

def draw_court(ax=None, color='black', lw=2, outer_lines=True):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

def scatter_plot_by_category(feat, df):
    alpha = 0.1
    gs = df.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)
        
        
#### Cleaning
        
        
        
        
####

        
df_raw = pd.read_csv("data.csv")
df = df_raw.copy()

df['secs_from_period_end']   = 60*df['minutes_remaining']+df['seconds_remaining']
df['secs_from_period_start'] = 60*(11-df['minutes_remaining'])+(60-df['seconds_remaining'])
df['secs_from_start']   = (df['period'] <= 4).astype(int)*(df['period']-1)*12*60 + (df['period'] > 4).astype(int)*((df['period']-4)*5*60 + 3*12*60) + df['secs_from_period_start']
df['dist'] = np.sqrt(df['loc_x']**2 + df['loc_y']**2)
df['away/home'] = df['matchup'].str.contains('vs').astype('int')
df['game_date'] = pd.to_datetime(df['game_date'])
df['game_year'] = df['game_date'].dt.year

loc_x_zero = (df['loc_x'] == 0)

df['angle'] = np.array([0]*len(df))
df['angle'][~loc_x_zero] = np.arctan(df['loc_y'][~loc_x_zero] / df['loc_x'][~loc_x_zero])
df['angle'][loc_x_zero] = np.pi / 2

to_drop = ['period', 'minutes_remaining', 'seconds_remaining', 'team_id', 'team_name', 'matchup', 'lon', 'lat',
           'game_id', 'game_event_id', 'game_date', 'playoffs', 'shot_distance', 'secs_from_period_start', 'season', 'game_date']

df.set_index('shot_id', inplace=True)

null_values = df['shot_made_flag'].isnull()

df = df.drop(to_drop, axis = 1)

under_10_secs = (df['secs_from_period_end'] < 10)==True
under_10 = [int(i) for i in under_10_secs]
last_10_bool = df[under_10_secs]
df['last_10s'] = under_10

categorical_cols = ['action_type', 'combined_shot_type', 'shot_type', 'shot_zone_area',
                        'shot_zone_basic', 'shot_zone_range', 'opponent']

for col in categorical_cols:
    dummies = pd.get_dummies(df[col])
    dummies = dummies.add_prefix("{}_".format(col))
    df.drop(col, axis=1, inplace=True)
    df = df.join(dummies)

#df_cont = df.drop(discrete_features, axis = 1)
    
    
    
plt.figure(figsize=(20,5))

# shot_zone_area
plt.subplot(131)
draw_court(); plt.ylim(-60,440); plt.xlim(270,-270)
scatter_plot_by_category('shot_zone_area', df_raw)
plt.title('shot_zone_area')

# shot_zone_basic
plt.subplot(132)
draw_court(); plt.ylim(-60,440); plt.xlim(270,-270)
scatter_plot_by_category('shot_zone_basic', df_raw)
plt.title('shot_zone_basic')

# shot_zone_range
plt.subplot(133)
draw_court(); plt.ylim(-60,440); plt.xlim(270,-270)
scatter_plot_by_category('shot_zone_range', df_raw)
plt.title('shot_zone_range')




y = df['shot_made_flag'].copy()

df = df.drop(['shot_made_flag'], axis = 1)
X = df[~null_values]
y = y[~null_values]

x_tr, x_test, y_tr, y_test = train_test_split(np.array(X_cont), np.array(y), test_size = 0.3)