# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 19:38:22 2023

@author: Think-pad
"""

import Soccer_IO as mio
import Soccer_Viz as mviz
import Soccer_Velocities as mvel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




DATADIR = '/Users/Think-pad/Downloads/Game1_2/Game1'

events = mio.read_event_data(DATADIR)

tracking_home = mio.tracking_data(DATADIR,'Home')
tracking_away = mio.tracking_data(DATADIR,'Away')

tracking_home =mio.to_metric_coordinates(tracking_home)
tracking_away =mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)

tracking_home,tracking_away,events = mio.to_single_playing_direction(tracking_home,tracking_away,events)

PLOTDIR = DATADIR
mviz.save_match_clip(tracking_home.iloc[1:1+500],tracking_away.iloc[1:1+500],PLOTDIR,fname = 'home_goal_2',include_player_velocities=False)

tracking_home = mvel.calc_player_velocities(tracking_home,smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away,smoothing=True)


mviz.plot_frame(tracking_home.loc[1000], tracking_away.loc[1000],include_player_velocities=True,annotate=True)

plt.show()
mviz.plot_events(events.loc[1:1], indicators = ['Marker','Arrow'],annotate=True)
plt.show()
