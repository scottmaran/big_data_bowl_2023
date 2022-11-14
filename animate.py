

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from IPython.display import HTML

''' 
Class to animate NFL Plays
'''
class AnimatePlay():
    
    # initialize variables we need. 
    # Example: Need to initialize the figure that the animation command will return
    def __init__(self, play_df) -> None:
        
        self.MAX_FIELD_PLAYERS = 22
        self.start_x = 0
        self.stop_x = 120
        self.start_y = 53.3
        self.play_df = play_df
        self.frames_list = play_df.frameId.unique()
        self.games_df = pd.read_csv("data/games.csv")
        
        fig, ax = plt.subplots(1, figsize=(8,4))
        
        self.fig = fig
        self.field_ax = ax
        
        # create new axis for home, away, jersey
        self.ax_home = self.field_ax.twinx()
        self.ax_away = self.field_ax.twinx()
        self.ax_jersey = self.field_ax.twinx()
        
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.frames_list),
                                          init_func=self.setup_plot, blit=False)
        
        plt.close()
        
    # initialization function for animation call
    def setup_plot(self):

        endzones = True
        linenumbers = True
        
        # set axis limits
        self.set_axis_plots(self.field_ax, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_home, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_away, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_jersey, self.stop_x, self.start_y)

        # set up colors and patches for field
        self.set_up_field(endzones, linenumbers)
        
        # create scatterplots on axis for data
        self.scat_field = self.field_ax.scatter([], [], s=50, color='orange')
        self.scat_home = self.ax_home.scatter([], [], s=50, color='blue')
        self.scat_away = self.ax_away.scatter([], [], s=50, color='red')
        
        # add direction stats and jersey numbers/names
        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._a_dir_list = []
        self._a_or_list = []
        for _ in range(self.MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'white'))
            self._scat_number_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            self._scat_name_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            
            self._a_dir_list.append(self.field_ax.add_patch(patches.Arrow(0, 0, 0, 0, color = 'k')))
            self._a_or_list.append(self.field_ax.add_patch(patches.Arrow(0, 0, 0, 0, color = 'k')))
        
        # return all axis plots that you want to update
        return (self.scat_field, self.scat_home, self.scat_away, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)
    
    def update(self, i):
        time_df = self.play_df.query("frameId == @i+1")
        
        #time_df['team_indicator'] = self.add_team_indicator(time_df)
        
        label_list = time_df.team.unique()
        label1= label_list[0]
        label2 = label_list[1]
        label3 = label_list[2]
        
        # update each team/football x,y coordinates
        for label in label_list:
            label_data = time_df[time_df.team == label]
            
            if label == label1:
                self.scat_field.set_offsets(label_data[['x','y']].to_numpy())
            elif label == label2:
                self.scat_home.set_offsets(label_data[['x','y']].to_numpy())
            elif label == label3:
                self.scat_away.set_offsets(label_data[['x','y']].to_numpy())
        
        #add direction and jersey info
        jersey_df = time_df[time_df.jerseyNumber.notnull()].reset_index()
        
        for (index, row) in jersey_df.iterrows():
            #self._scat_jersey_list[index].set_position((row.x, row.y))
            #self._scat_jersey_list[index].set_text(row.position)
            self._scat_number_list[index].set_position((row.x, row.y+1.9))
            self._scat_number_list[index].set_text(int(row.jerseyNumber))
            #self._scat_name_list[index].set_position((row.x, row.y-1.9))
            #self._scat_name_list[index].set_text(row.displayName.split()[-1])
            
            player_orientation_rad = self.deg_to_rad(self.convert_orientation(row.o))
            player_direction_rad = self.deg_to_rad(self.convert_orientation(row.dir))
            player_speed = row.s
            
            player_vel = np.array([np.real(self.polar_to_z(player_speed, player_direction_rad)), np.imag(self.polar_to_z(player_speed, player_direction_rad))])
            player_orient = np.array([np.real(self.polar_to_z(2, player_orientation_rad)), np.imag(self.polar_to_z(2, player_orientation_rad))])
            
            self._a_dir_list[index].remove()
            self._a_dir_list[index] = self.field_ax.add_patch(patches.Arrow(row.x, row.y, player_vel[0], player_vel[1], color = 'k'))
            
            self._a_or_list[index].remove()
            self._a_or_list[index] = self.field_ax.add_patch(patches.Arrow(row.x, row.y, player_orient[0], player_orient[1], color = 'grey', width = 2))
                

        return (self.scat_field, self.scat_home, self.scat_away, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)
    
    def set_up_field(self, endzones=True, linenumbers=True) -> None:
        yard_numbers_size = self.fig.get_size_inches()[0]*1.5

        # color field 
        rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                                    edgecolor='r', facecolor='darkgreen', zorder=0)
        self.field_ax.add_patch(rect)

        # plot
        self.field_ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
                    80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
                    [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
                    53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
                    color='white')

        # Endzones
        if endzones:
            ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
            ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
            self.field_ax.add_patch(ez1)
            self.field_ax.add_patch(ez2)
            
        if endzones:
            hash_range = range(11, 110)
        else:
            hash_range = range(1, 120)

        # add hashes
        for x in hash_range:
            self.field_ax.plot([x, x], [0.4, 0.7], color='white')
            self.field_ax.plot([x, x], [53.0, 52.5], color='white')
            self.field_ax.plot([x, x], [22.91, 23.57], color='white')
            self.field_ax.plot([x, x], [29.73, 30.39], color='white')
            
        # add linenumbers
        if linenumbers:
                for x in range(20, 110, 10):
                    numb = x
                    if x > 50:
                        numb = 120 - x
                    self.field_ax.text(x, 5, str(numb - 10),
                            horizontalalignment='center',
                            fontsize=yard_numbers_size,  # fontname='Arial',
                            color='white')
                    self.field_ax.text(x - 0.95, 53.3 - 5, str(numb - 10),
                            horizontalalignment='center',
                            fontsize=yard_numbers_size,  # fontname='Arial',
                            color='white', rotation=180)

        self.field_ax.set_xlim(self.start_x, self.stop_x)
        self.field_ax.set_ylim(0, self.start_y)
        self.field_ax.set_xticks(range(self.start_x,self.stop_x, 10))

    @staticmethod
    def set_axis_plots(ax, max_x, max_y) -> None:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])
        
    @staticmethod
    def convert_orientation(x):
        return (-x + 90)%360
    
    @staticmethod
    def polar_to_z(r, theta):
        return r * np.exp( 1j * theta)
    
    @staticmethod
    def deg_to_rad(deg):
        return deg*np.pi/180
        
    # returns column of team indicator
    def add_team_indicator(self, play_df, game_id=None):
        
        if game_id == None:
            game_id = play_df.gameId.values[0]
        
        game_df = self.games_df.query("gameId == @game_id")
        home_team = game_df['homeTeamAbbr'].values[0]
        away_team = game_df['visitorTeamAbbr'].values[0]
        
        conditions = [
            (play_df.team == home_team),
            (play_df.team == away_team),
            (play_df.team == 'football')
        ]
        choices = [1,2,3]
        return np.select(conditions, choices)