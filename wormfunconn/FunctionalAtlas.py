import numpy as np
import wormfunconn as wfc
import os,pickle

class FunctionalAtlas:
    
    fname = "functionalatlas.pickle"
    
    strain = "wild type"
    
    def __init__(self,neu_ids,params):
        '''Temporary simplified constructor for mock dataset.
        
        Parameters
        ----------
        neu_ids: list of str
            List of the neuronal IDs.
        params: (N, N, M) array_like
            M parameters of the response functions of the NxN connections.
        '''
        
        self.neu_ids = np.array(neu_ids)
        self.params = np.array(params)
        
        self.n_neu = self.neu_ids.shape[0]
        
        self.strain = "wild type"
        
    @classmethod
    def from_file(cls,folder,fname=None):
        '''Load the pickled version of a FunAtlas object.
        
        Parameters
        ----------
        folder: string
            Folder containing the recording.
        fname: string (optional)
            File name of the pickled object. If None, the default file name is 
            used. Default: None.
            
        Returns
        -------
        inst: Fconn 
            Instance of the class.
        '''
    
        if fname is None: 
            fname = cls.fname
        else: 
            if fname.split(".")[-1] != "pickle": fname += ".pickle"
        
        if os.path.isfile(folder+fname):
            f = open(folder+fname,"rb")
            inst = pickle.load(f)
            f.close()
            return inst
        else:
            raise Exception("Pickled file not found.")
    
    def to_file(self,folder,fname):
        '''Saves pickled version of the object to file.
        
        Parameters
        ----------
        folder: string (optional)
            Folder where to save the file. If None, the function uses the folder
            specified at instantiation. Default: None.
        fname: string (optional)
            Destination file name. If None, the function uses the default file
            name of the class. Default: None.
        '''        
        
        if folder is None:
            folder = self.folder
        else:
            self.folder = folder
        
        if fname is not None:
            if fname.split(".")[-1] != "pickle": fname += ".pickle"
            self.fname = fname

        pickle_file = open(self.folder+self.fname,"wb")
        pickle.dump(self,pickle_file)
        pickle_file.close()
        
    def get_strain(self):
        try:
            return self.strain
        except:
            return "wild type"
        
    def get_responses(self, stimulus, dt, stim_neu_id, resp_neu_ids=None,
                      threshold=0.0, top_n=None, sort_by_amplitude=True):
        '''Compute neural responses given a stimulus.
       
        Parameters
        ----------
        stimulus: (M,) array_like
            Input stimulus applied to the stimulated neuron.
        dt: float
            Time step.
        stim_neu_id: str
            ID of the stimulated neuron.
        resp_neu_ids: str, list of str
            IDs of the neurons of which to calculate the responses. If None,
            all the responses above a threshold are returned. Default: None.
        threshold: float
            Some kind of threshold.
        top_n: int (optional)
            If not None, the function will return the top_n responses with the
            largest absolute peak amplitude.
        sort_by_amplitude: bool (optional)
            Whether to sort the output curves by amplitude. Default: True.
           
        Returns
        -------
        responses: (N, M) numpy.ndarray
            Responses of the N responding neurons, for M timesteps (same length
            as the input stimulus).
        labels: list of str
            IDs of the responding neurons.
        msg: str
            Message containing comments to output.
        '''
        
        # Initialize the output message
        msg = ""
        
        # Find index of stimulated neuron from its ID
        sn_i = np.where(self.neu_ids==stim_neu_id)[0]
        if len(sn_i)>0: sn_i = sn_i[0]
        else: return None
        
        if resp_neu_ids is None:
            # Return either:
            # 1) The top_n absolute-peak-amplitude responses, if top_n is not 
            #    None.
            # 2) The responses that pass the threshold, if top_n is None.
            # In both cases compute the responses for all the neurons and 
            # select them before returning.
            resp_neu_ids = self.neu_ids
            resp_neu_ids_was_None = True
        else:   
            resp_neu_ids_was_None = False
            # If a single ID was passed, make it a list
            try: 
                len(resp_neu_ids)
                was_scalar = False
            except: 
                resp_neu_ids = [resp_neu_ids]
                was_scalar = True
            
        # Make array of times
        n_t = len(stimulus)
        t = np.arange(n_t)*dt
            
        # Initialize output array
        n_resp = len(resp_neu_ids)
        # Keep out a list instead of initializing an array so that you can
        # leave out neurons for which we don't have data, instead of returning
        # nans, which mess with the sorting.
        out = [] #np.zeros((n_resp,n_t))
         
        # Compute output
        no_data_for = [] # Keep track of neurons that are not in the dataset.
        labels = [] # As you go through them, make the list of IDs. 
        for i in np.arange(n_resp):
            rn_id = resp_neu_ids[i]
            # Find responding neuron's index from ID
            rn_i = np.where(self.neu_ids==rn_id)[0]
            if len(rn_i)>0: 
                # The ID was found.
                rn_i = rn_i[0]
                labels.append(self.neu_ids[rn_i])
            
                if rn_i == sn_i:
                    # If the activity of the stimulated neuron is requested,
                    # return the stimulus.
                    out.append(stimulus)
                    msg += "The activity of the stimulated neuron ("+\
                           stim_neu_id+") is the "+\
                           "activity set as stimulus. "
                elif self.params[rn_i,sn_i] is not None:
                    # If there are parameters for this connection,
                    # compute mock response function
                    rf = wfc.exp_conv_2b(t,self.params[rn_i,sn_i])
                    
                    # Convolve with stimulus
                    out.append(wfc.convolution(rf,stimulus,dt,8))
                else:
                    # If we don't have data for this connection, don't pass
                    # anything. Passing nans messes with the sorting.
                    no_data_for.append(i)
                    labels.pop(-1)
                    #out[i] = np.nan
                
            else:
                # The ID was not found. Don't pass anything. Passing nans
                # messes with the sorting.
                #labels.append("")
                msg += "'"+rn_id+"' is not a neuron. "
        
        out = np.array(out)        
        labels = np.array(labels)
        
        # Compile a message listing the neurons for which there is no data.
        if len(no_data_for)>0:
            msg += "These neurons are not in the dataset: "
            for ndf in no_data_for:
                msg += self.neu_ids[i]+","
            # Replace the last comma with period.
            msg = msg[:-1]
            msg += ". "
        
        # If either top_n or the threshold must be used, select from the array
        # of all the responses that was created above.
        if resp_neu_ids_was_None and top_n is not None:
            # Select the top_n responses in terms of max(abs())
            outsort = np.argsort(np.max(np.abs(out),axis=-1))[::-1]
            out = out[outsort[:top_n+1]]
            labels = labels[outsort[:top_n+1]]
        elif resp_neu_ids_was_None and top_n is None:
            # Select the responses that pass the threshold.
            outselect = np.where(np.max(np.abs(out),axis=-1)>=threshold)[0]
            out = out[outselect]
            labels = labels[outselect]
            
        # Sort the responses by amplitude. Not strictly needed if top_n is 
        # not None (because out has been sorted above), but whatever.
        if sort_by_amplitude:
            outsort = np.argsort(np.max(np.abs(out),axis=-1))[::-1]
            out = out[outsort]
            labels = labels[outsort]
            
            # Also add a (i) to the labels, indicating the rank of that
            # response.
            # Make a new list to automatically redefine the <Un dtype.
            labels_new = [] 
            for p in np.arange(len(labels)):
                labels_new.append(labels[p]+" ("+str(p)+")")
            
            labels = np.array(labels_new)
            
        confidences = self.get_confidences(stim_neu_id,resp_neu_ids)
        
        return out, labels, confidences, msg
        
    def get_confidences(self,stim_neu_id,resp_neu_ids):
        # Temporary, for mock dataset
        return np.ones(len(resp_neu_ids),dtype=float)
        
    def get_scalar_connectivity(self, mode="amplitude", 
        threshold={"amplitude":0.1}, return_all = True, dtype=int):
        '''Get scalar version of functional connectivity (amplitude, timescales,
        or other.
        
        Parameters
        ----------
        mode: str (optional)
            Type of scalar quantity to extract from functional connectivity.
            Can be amplitude, timescales. Default: amplitude.
            
        threshold: dict (optional)
            Specify the thresholds, in a dictionary, determining which 
            connections are significant. Currently only implemented for 
            amplitude, but other future thresholds could include the timescales.
            Default: {\"amplitude\":0.8}.
            
        dtype: dtype (optional)
            If int, the connectivity is multiplied by 100 and returned as an
            integer.
            
        Returns
        -------
        s_fconn: numpy.ndarray
            A scalar version of the functional connectivity.
        ann: numpy.ndarray of object
            Annotations for the connections (functionally stable, functionally
            multistable, functionally variable). Empty for non-connected edges.
        '''
        
        folder = os.path.join(os.path.dirname(__file__), "../atlas/")
        i_map = np.loadtxt(folder+"funatlas_intensity_map.txt")
        
        i_map[np.isnan(i_map)] = 0
        i_map[np.isinf(i_map)] = 0
        
        i_map[np.abs(i_map)<threshold["amplitude"]] = 0
        if dtype==int:
            s_fconn = (i_map*100).astype(np.int32)
        else:
            s_fcon = i_map
        
        '''
        
        # Determine which connections are above the threshold.
        sel_conn = np.ones((self.n_neu,self.n_neu),dtype=np.bool)
        for key in threshold.keys():
            th = threshold[key]
            if key == "amplitude":
                sel_conn *= np.absolute(self.params[...,0])>th
        
        # Based on mode, select the parameters to extract.
        if mode=="amplitude":
            s_fconn = self.params[...,0]
        elif mode=="timescale_1":
            s_fconn = self.params[...,1]
        else:
            s_fconn = np.zeros((self.n_neu,self.n_neu))
        
        # Set to zero the entries for all the connections below the threshold.    
        s_fconn[~sel_conn] = 0.0
        '''
        # Get the annotation of the edges (functionally stable, functionally
        # multistable, functionally variable). For now, use some random 
        # assignment.
        ann = np.empty((self.n_neu,self.n_neu),dtype=object)
        ann[:] = ""
        '''for i in np.arange(self.n_neu):
            for j in np.arange(self.n_neu):
                if sel_conn[i,j]:
                    if i%2==0:
                        ann[i,j] = "functionally variable"
                    else:
                        ann[i,j] = "functionally stable"
                else:
                    ann[i,j] = ""
        '''
        if return_all:
            return s_fconn, ann
        else:
            return s_fconn
    
    def convert_s_fconn_to_table(self,s_fconn,ann,ds_type="head"):
        '''Converts the output of FunctionalAtlas.get_scalar_connectivity()
        to a table-like list of dictionaries with keys from, to, strength,
        ds_type, annotation.
        
        Parameters
        ----------
        s_fconn: numpy.ndarray
            Static functional connectivity matrix returned by 
            FunctionalAtlas.get_scalar_connectivity()
        ann: numpy.ndarray
            Annotations of the connections.
        ds_type: str (optional)
            Type of dataset. Default: head.
            
        Returns
        -------
        entries: list of dictionaries
            List of dictionaries with keys from, to, strength, ds_type, 
            annotation. Entries are present only for the connections that have
            strength different from zero. A threshold can be set in 
            FunctionalAtlas.get_scalar_connectivity() when getting s_fconn and
            ann.
        '''
        
        entries = []
        for i in np.arange(self.n_neu):
            for j in np.arange(self.n_neu):
                if s_fconn[i,j]!=0:
                    to_id = self.neu_ids[i]
                    from_id = self.neu_ids[j]
                    entry = {"from": from_id,
                             "to": to_id,
                             "strength": s_fconn[i,j],
                             "ds_type": ds_type,
                             "annotation": ann[i,j]}
                    entries.append(entry)
                    
        return entries
        
    @classmethod    
    def get_standard_stimulus(cls,nt,t_max=None,dt=None,stim_type="rectangular",
                              *args,**kwargs):
        '''Returns a stimulus from a set of standard stimuli. 
        
        Parameters
        ----------
        nt: int
            Number of time points.
        t_max: float (optional)
            Maximum time. If None, it will be determined from nt and dt. Either
            t_max or dt must be not None. Default: None.
        dt: float (optional)
            Time step. If None, it will be determined from nt and t_max. 
            Default: None.
        stim_type: str (optional)
            Type of stimulus. Can be rectangular, . Default: rectangular
            
        Returns
        -------
        stim: (N,) numpy.ndarray
            Stimulus.
        '''
        
        if t_max is None and dt is None:
            raise ValueError("Either t_max or dt must be not None.")
        elif dt is None:
            dt = t_max/nt
        
        if stim_type=="rectangular":
            stim = cls.stim_rectangular(nt,dt,*args,**kwargs)
        elif stim_type=="sinusoidal":
            stim = cls.stim_sinusoidal(nt,dt,*args,**kwargs)
        elif stim_type=="delta":
            stim = cls.stim_rectangular(nt,dt,duration=dt)
        elif stim_type =="realistic":
            stim = cls.stim_realistic(nt,dt,*args,**kwargs)
            
        return stim
        
    @staticmethod
    def get_standard_stim_kwargs(stim_type):
        if stim_type=="rectangular":
            kwargs = [{"name": "duration", "type": "float", "default": 1.0,
                        "label": "Duration (s)", "range": [0.,None]}]
        elif stim_type=="sinusoidal":
            kwargs = [{"name": "frequency", "type": "float", "default": 0.25,
                        "label": "Frequency (Hz)", "range": [0.,0.25]},
                       {"name": "phi0", "type": "float", "default": 0.0,
                        "label": "Phase", "range": [0.,6.28]}]
        elif stim_type=="delta":
            kwargs = []
        elif stim_type =="realistic":
            kwargs = [{"name": "tau1", "type": "float", "default": 1.0,
                        "label": "Timescale 1 (s)", "range": [0.5,100.]},
                       {"name": "tau2", "type": "float", "default": 0.8,
                        "label": "Timescale 2 (s)", "range": [0.5,100.]}]
                        
        return kwargs
   
    @staticmethod    
    def stim_rectangular(nt,dt,duration=1.):
        '''Rectangular stimulus.'''
        stim = np.zeros(nt)
        i1 = int(duration/dt)
        i1 = min(nt,i1) # Cannot be past the end of the array
        
        stim[:i1] = 1.
        return stim
    
    @staticmethod
    def stim_realistic(nt,dt,tau1=1.,tau2=0.8):
        '''Realistic stimulus, i.e. a convolution of two exponentials.
        '''
        t = np.arange(nt)*dt
        ec = wfc.ExponentialConvolution([1./tau1,1./tau2])
        stim = ec.eval(t)
        
        #p = 1,1/tau1,1/tau2-1/tau1
        #stim = wfc.exp_conv_2b(t,p)

        return stim
        
    @staticmethod
    def stim_sinusoidal(nt,dt,frequency=1.,phi0=0.):
        '''Sinusoidal stimulus.'''
        t = np.arange(nt)*dt
        stim = np.sin(2.*np.pi*frequency*t+phi0)
        
        return stim
        
    @staticmethod
    def get_code_snippet(nt,dt,stim_type,stim_kwargs,stim_neu_id,
                         resp_neu_ids=None,threshold=0.0,top_n=None,
                         sort_by_amplitude=True):
                         
        if len(resp_neu_ids)==0: resp_neu_ids = None
        
        if top_n=="None": top_n = None
        
        sn = "import numpy as np, matplotlib.pyplot as plt, os\n"+\
             "import wormfunconn as wfc\n"+\
             "\n"+\
             "# Get atlas folder\n"+\
             "folder = os.path.join(os.path.dirname(__file__),\"../atlas/\")\n"+\
             "\n"+\
             "# Create FunctionalAtlas instance from file\n"+\
             "funatlas = wfc.FunctionalAtlas.from_file(folder,\"mock\")\n"+\
             "\n"+\
             "# Set time-axis properties\n"+\
             "nt = 1000\n"+\
             "t_max = 100.0\n"+\
             "dt = t_max / nt\n"+\
             "\n"+\
             "stim = funatlas.get_standard_stimulus(nt,dt=dt,stim_type=\""+stim_type+"\","+\
             ','.join('{0}={1!r}'.format(k,v) for k,v in stim_kwargs.items())+")\n"+\
             "\n"+\
             "stim_neu_id = \""+stim_neu_id+"\"\n"+\
             "resp_neu_ids = "+str(resp_neu_ids)+"\n"+\
             "top_n = "+str(top_n)+"\n"+\
             "threshold = "+str(threshold)+"\n"+\
             "sort_by_amplitude = "+str(sort_by_amplitude)+"\n"+\
             "resp, labels, msg = funatlas.get_responses(stim, dt, stim_neu_id, resp_neu_ids=resp_neu_ids, top_n=top_n, threshold=threshold, sort_by_amplitude=sort_by_amplitude)\n"+\
             "\n"+\
             "print(msg)\n"+\
             "\n"+\
             "time = np.arange(nt)*dt\n"+\
             "for i in np.arange(resp.shape[0]):\n"+\
             "\tplt.plot(time,resp[i],label=labels[i],,alpha=confidences[i])\n"+\
             "plt.legend()\n"+\
             "plt.show()"
             
        return sn
             

