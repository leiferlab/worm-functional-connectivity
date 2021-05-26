import numpy as np
import matplotlib.pyplot as plt
import wormfunconn as wfc
import os

# Get atlas folder
folder = os.path.join(os.path.dirname(__file__),"../atlas/")

# Create FunctionalAtlas instance from file
funatlas = wfc.FunctionalAtlas.from_file(folder,"mock")

# Generate the stimulus array
nt = 1000 # Number of time points 
dt = 0.1 # Time step
stim_type="rectangular" # Type of standard stimulus
dur = 2. # Duration of the stimulus
stim = funatlas.get_standard_stimulus(nt,dt,stim_type=stim_type,duration=dur)

# Get the responses
stim_neu_id = "AVAL"
resp_neu_ids = ["AVAR","ASEL","AWAL"]
resp = funatlas.get_responses(stim, dt, stim_neu_id, resp_neu_ids)

# Plot
plt.plot(resp.T)
plt.show()
