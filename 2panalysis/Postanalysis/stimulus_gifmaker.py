
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter 
import STRF_utils as RF

stimulus_dir = os.path.abspath('C:\\Users\\vargasju\\PhD\\experiments\\2p\\T4T5_STRF_glucla_rescues\\stimuli_arrays')
stim_type = '10max_5degbox'
stimulus = RF.load_stimulus(stimulus_dir,stim_type)
stimulus = stimulus[:300,:,:]

# Create the figure and axes objects
fig, ax = plt.subplots()

# Set the initial image
im = ax.imshow(stimulus[0], animated=True,cmap='Greys')
ax.axis('off')
def update(i):
    im.set_array(stimulus[i])
    return im, 

# Create the animation object
animation_fig = animation.FuncAnimation(fig, update, frames=len(stimulus), interval=50, blit=True,repeat_delay=10,)

# Show the animation
#plt.show()

animation_fig.save(os.path.join(stimulus_dir,"stim_%s.gif"%(stim_type)), writer=PillowWriter(fps=20))
pepe