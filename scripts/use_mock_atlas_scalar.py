import numpy as np
import matplotlib.pyplot as plt
import wormfunconn as wfc
import os

# Get atlas folder
folder = os.path.join(os.path.dirname(__file__),"../atlas/")

# Create FunctionalAtlas instance from file
funatlas = wfc.FunctionalAtlas.from_file(folder,"mock")

# Get scalar version of functional connectivity
s_fconn = funatlas.get_scalar_connectivity(mode="amplitude",threshold={"amplitude":1.2})

# Plot
plt.imshow(s_fconn,cmap="coolwarm",vmin=-1,vmax=1)
plt.show()
