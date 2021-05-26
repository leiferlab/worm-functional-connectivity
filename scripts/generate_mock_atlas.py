import numpy as np
import wormfunconn as wfc
import os

# Load the neural IDs
folder = "/".join(wfc.__file__.split("/")[:-1])+"/"
f = open(folder+"aconnectome_ids.txt")
lines = f.readlines()
f.close()

neu_ids = [l.split("\t")[1][:-1] for l in lines]

# Generate sample parameters
n_neu = len(neu_ids)
params = np.zeros((n_neu,n_neu,3))
params[...,0] = 1.
params[...,1] = np.arange(n_neu)*0.01+0.1
params[...,1] *= (np.arange(n_neu)*0.01+0.1)[:,None]
params[...,2] = np.arange(n_neu)*0.001+0.05

funatlas = wfc.FunctionalAtlas(neu_ids,params)

dst_folder = os.path.join(os.path.dirname(__file__),"../atlas/")
funatlas.to_file(dst_folder,"mock")
