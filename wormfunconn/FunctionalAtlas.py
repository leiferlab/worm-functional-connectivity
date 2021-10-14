import numpy as np
import wormfunconn as wfc
import os,pickle

class FunctionalAtlas:
    
    fname = "functionalatlas.pickle"
    
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
        
    def get_responses(self, stimulus, dt, stim_neu_id, resp_neu_ids=None,
                      threshold=0.0):
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
           
        Returns
        -------
        responses: (N, M) numpy.ndarray
            Responses of the N responding neurons, for M timesteps (same length
            as the input stimulus).
        out_neu_ids: list of str
            IDs of the responding neurons.
        '''
        
        # Find index of stimulated neuron from its ID
        sn_i = np.where(self.neu_ids==stim_neu_id)[0]
        if len(sn_i)>0: sn_i = sn_i[0]
        else: return None
        
        if resp_neu_ids is None:
            # Temporarily use the first N=threshold neurons in the list
            threshold = int(threshold)
            resp_neu_ids = self.neu_ids[:threshold]
        else:   
            # If a single ID was passed, make it a list
            try: len(resp_neu_ids)
            except: resp_neu_ids = [resp_neu_ids]
            
        # Make array of times
        n_t = len(stimulus)
        t = np.arange(n_t)*dt
            
        # Initialize output array
        n_resp = len(resp_neu_ids)
        out = np.zeros((n_resp,n_t))
         
        # Compute output
        for i in np.arange(n_resp):
            rn_id = resp_neu_ids[i]
            # Find responding neuron's index from ID
            rn_i = np.where(self.neu_ids==rn_id)[0]
            if len(rn_i)>0: rn_i = rn_i[0]
            else: out[i] = np.nan
                
            # Compute mock response function
            rf = wfc.exp_conv_2b(t,self.params[rn_i,sn_i])
            
            # Convolve with stimulus
            out[i] = wfc.convolution(rf,stimulus,dt,8)
            
        return out
        
    def get_scalar_connectivity(self, mode="amplitude", threshold={"amplitude":0.8}):
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
            
        Returns
        -------
        s_fconn: numpy.ndarray
            A scalar version of the functional connectivity.
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
        
        return s_fconn
    
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
                        "label": "Duration (s)"}]
        elif stim_type=="sinusoidal":
            kwargs = [{"name": "frequency", "type": "float", "default": 1.0,
                        "label": "Frequency (Hz)"},
                       {"name": "phi0", "type": "float", "default": 0.0,
                        "label": "Phase"}]
        elif stim_type=="delta":
            kwargs = []
        elif stim_type =="realistic":
            kwargs = [{"name": "tau1", "type": "float", "default": 1.0,
                        "label": "Timescale 1 (s)"},
                       {"name": "tau2", "type": "float", "default": 0.8,
                        "label": "Timescale 2 (s)"}]
                        
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
        p = 1,1/tau1,1/tau2-1/tau1
        stim = wfc.exp_conv_2b(t,p)

        return stim
        
    @staticmethod
    def stim_sinusoidal(nt,dt,frequency=1.,phi0=0.):
        '''Sinusoidal stimulus.'''
        t = np.arange(nt)*dt
        stim = np.sin(2.*np.pi*frequency*t+phi0)
        
        return stim
