import numpy as np
from scipy.optimize import root as root
from scipy.optimize import root_scalar as root_scalar
import pumpprobe as pp

class ExponentialConvolution:
    
    def __init__(self,gs,A=1.):
        self.n_branches = 1
        g1 = gs[0]
        exp1 = {"g":g1,"factor":g1*A,"power_t":0,"branch":0}
        self.exp = [[exp1]]
        self.steps = [{"type": "init", "g":g1, "A":A, "branch": 0}]
        
        for g in gs[1:]:
            self.convolve_exp(g)
    
    @classmethod
    def from_steps(cls,steps):
        '''Creates an instance of ExponentialConvolution given a list of 
        dictionaries representing each convolution step. The first step must be
        of type "init" {"type": "init", "A": float, "g": float}. The following
        steps can be either of type "convolve" 
        {"type": "convolve", "g": float, "branch": int}
        or of type "branch_path"
        {"type": "branch_path", "A": float, "g": float}
        
        Parameters
        ----------
        steps: list of dictionaries
            List of dictionaries representing the convolution steps. See above
            for requirements.
        
        Returns
        -------
        inst: ExponentialConvolution
            Instance of ExponentialConvolution.      
        '''
        
        assert steps[0]["type"]=="init"
        inst = cls([steps[0]["g"]],steps[0]["A"])
        for step in steps:
            if step["type"]=="convolve":
                inst.convolve_exp(step["g"],step["branch"])
            elif step["type"]=="branch_path":
                inst.branch_path(step["g"],step["A"])
                
        return inst
        
    @classmethod
    def from_dict(cls,diz):
        params = diz["params"]
        n_branches = diz["n_branches"]
        n_branch_params = diz["n_branch_params"]
        
        # Back-compatibility with a wrong default being set
        if type(n_branch_params)==int: n_branch_params = [n_branch_params]
        
        inst = cls([params[1]],params[0])
        for ip in np.arange(n_branch_params[0])[2:]:
            inst.convolve_exp(params[ip],branch=0)
        
        prev_par = np.cumsum(n_branch_params)
        for ib in np.arange(n_branches)[1:]:
            pb = params[prev_par[ib-1]:]
            inst.branch_path(pb[1],pb[0])
            for ip in np.arange(n_branch_params[ib])[2:]:
                inst.convolve_exp(pb[ip],branch=ib)
                
        return inst

    def convolve_exp(self,g,branch=-1):
        '''Convolve an exponential to the existing symbolic object, either on
        a specific branch, or on all the branches. If the object has multiple
        branches, the latter case can be seen as a rejoining of the branches.
        
        Parameters
        ----------
        g: float
            Gamma of the exponential to be convolved.
        branch: int (optional)
            Index of the branch to which to apply the convolution. If -1, the
            convolution is applied to all the branches. Default: -1.
        
        '''
        self.steps.append({"type": "convolve", "g": g, "branch": branch})
        expi = []
        for prev_exp in self.exp[-1]:
            old_g = prev_exp["g"]
            old_factor = prev_exp["factor"]
            old_power_t = prev_exp["power_t"]
            old_branch = prev_exp["branch"]
            if branch!=-1 and old_branch!=branch:
                expi.append(prev_exp)
                continue
            gji = g-old_g
            
            found_old_g = False
            found_g = False
            old_g_idx = None
            g_idx = None
            
            if old_g!=g and old_power_t==0: 
                # If I have an exp(-ax) convolved with exp(-bx)
                for i in np.arange(len(expi)):
                    if expi[i]["g"]==old_g and expi[i]["branch"]==old_branch and expi[i]["power_t"]==0: 
                        old_g_idx = i; found_old_g = True
                    if expi[i]["g"]==g and expi[i]["branch"]==old_branch and expi[i]["power_t"]==0:
                        g_idx = i; found_g = True
                    if found_old_g and found_g: break
                
                if old_g_idx is None:
                    expi.append({"g":old_g,"factor":old_factor*g/gji,"power_t":0,"branch":old_branch})
                else:
                    expi[old_g_idx]["factor"] += old_factor*g/gji
                if g_idx is None:
                    expi.append({"g":g,"factor":-old_factor*g/gji,"power_t":0,"branch":old_branch})
                else:
                    expi[g_idx]["factor"] += -old_factor*g/gji
                    
            elif old_g==g:
                # If I have an x^n*exp(-ax) convolved with exp(-ax)
                expi.append({"g":g,
                             "factor":old_factor*g/(old_power_t+1),
                             "power_t":old_power_t+1,"branch":old_branch})
                    
            elif old_g!=g and old_power_t==1:
                # If I have an x*exp(-ax) convolved with exp(-bx)
                expi.append({"g":old_g,
                             "factor": old_factor*g/(g-old_g),
                             "power_t": 1,"branch":old_branch})
                expi.append({"g":old_g,
                             "factor": -old_factor*g/np.power(old_g-g,2),
                             "power_t": 0,"branch":old_branch})  
                expi.append({"g":g,
                             "factor": old_factor*g/np.power(old_g-g,2),
                             "power_t": 0,"branch":old_branch}) 
                             
                             
            elif old_g!=g and old_power_t>1:
                # If I have an x^n*exp(-ax) convolved with exp(-bx)
                sign = +1
                mult = old_factor*g/(g-old_g)
                for pot in np.arange(old_power_t+1)[::-1]:
                    expi.append({"g":old_g,"factor":mult,"power_t":pot,"branch":old_branch})
                    mult *= -pot/(g-old_g)
                
                expi.append({"g":g,
                             "factor": old_factor*g*((-1)**(old_power_t+1))*np.math.factorial(old_power_t)/np.power(g-old_g,old_power_t+1),
                             "power_t": 0,"branch":old_branch})
                             
        self.exp.append(expi)
        
    def branch_path(self,g,A):
        '''Branch the path of convolutions.
        
        Parameters
        ----------
        g: float
            gamma of the first exponential of the new branch.
        A: float
            Initial overall amplitude of the new branch.
        '''
        self.exp[-1].append({"g":g,
                             "factor":g*A,
                             "power_t": 0,
                             "branch": self.n_branches})
        self.steps.append({"type":"branch_path",
                           "g":g,
                           "A":A,
                           "branch":self.n_branches})
        self.n_branches += 1
        
    def eval(self,x,dtype=np.float64,drop_branches=None):
        '''Evaluates the ExponentialConvolution in the time domain.
        
        Parameters
        ----------
        x: array_like
            Time axis. All times should be positive.
        dtype: type (optional)
            Type of the output array. Default: np.float64        
        drop_branches: int or array_like of int
            Branches to be ignored in the evaluation. Default: None.
            
        Returns
        -------
        out: numpy.ndarray
            ExponentialConvolution evaluated on x.
        '''
        assert np.all(x>=0)
        
        if drop_branches is not None:
            try: len(drop_branches)
            except: drop_branches = [drop_branches]
        
        out = np.zeros_like(x,dtype=dtype)
            
        for exp in self.exp[-1]:
            # Skip terms that are in excluded branches
            branch = exp["branch"]
            if drop_branches is not None:
                if branch in drop_branches: continue
                
            g = exp["g"]
            factor = exp["factor"]
            power_t = exp["power_t"]
            
            if power_t==0: mult=1.
            else: mult=np.power(x,power_t)
            out += factor*mult*np.exp(-g*x)
            
        return out
    
    @staticmethod
    def eval2(x,gs,cs,pt,dtype=np.float64):
        '''Static alternative evaluation method that takes the parameters of
        ExponentialConvolution as inputs (obtainable from 
        ExponentialConvolution.get_bare_params()).
        
        Parameters
        ----------
        x: array_like
            Time axis. All times should be positive.
        gs: array_like of floats
            Gammas
        cs: array_like of floats
            Amplitudes
        pt: array_like of floats
            Powers of t
            
        Returns
        -------
        out: numpy.ndarray
            ExponentialConvolution evaluated on x.
        '''
        assert np.all(x>=0)
        out = np.zeros_like(x,dtype=dtype)
            
        for i in np.arange(len(gs)):
            g = gs[i]
            factor = cs[i]
            power_t = pt[i]
            if power_t==0: mult=1.
            else: mult=np.power(x,power_t)
            out += factor*mult*np.exp(-g*x)
            
        return out
        
    def eval_ft(self,omega):
        '''Evaluates the ExponentialConvolution in the frequency domain.
        
        Parameters
        ----------
        omega: array_like
            Frequency axis.
            
        Returns
        -------
        out: numpy.ndarray of np.complex128
            ExponentialConvolution evaluated on omega.
        '''
        
        out = np.zeros_like(omega,dtype=np.complex128)
        
        for exp in self.exp[-1]:
            g = exp["g"]
            a = exp["factor"]
            n = exp["power_t"]
            
            out+=a*(-1)**(n+1)*np.math.factorial(n)/( (1.0j*omega-g)**(n+1) )

        return out
        
    def get_derivative_t_0(self):
        '''Return the derivative at time=0
        '''
        d = 0
        for e in self.exp[-1]:
            # The derivative of each c t^p e^-gt is 
            # c p (0+)^(p-1) - c g (0+)^p.
            # So each term contributes a nonzero amount to the derivative at 
            # time 0 only if p=0 or p-1=0
            if e["power_t"]==0:
                d-=e["factor"]*e["g"]
            elif e["power_t"]==1:
                d+=e["factor"]
        
        if d == 0: print("derivative of EC = 0", self.exp[-1])
        
        return d
        
    def get_derivative2_t_0(self,dt=1e-1):
        '''Return the derivative at time=0. Numerically, when there are weird
        cancellations that don't work. Single time step, beware of accuracy.
        '''
        y1 = 0 
        i = 1
        while abs(y1) < 1e-6:
            y1 = self.eval(dt*i)
            i+=1
        y0 = self.eval(0)
        d = (y1-y0)/dt
                
        return d
        
    def get_bare_params(self):
        '''Returns arrays of the parameters for each individual term in 
        ExponentialConvolution.
        
        Returns
        -------
        gs: array_like of floats
            Gammas
        cs: array_like of floats
            Amplitudes
        pt: array_like of floats
            Powers of t
        bs: array_like of ints
            Branch index
        '''
        gs = np.zeros(len(self.exp[-1]))
        cs = np.zeros(len(self.exp[-1]))
        pt = np.zeros(len(self.exp[-1]))
        bs = np.zeros(len(self.exp[-1]))
        for i in np.arange(len(self.exp[-1])):
            gs[i] = self.exp[-1][i]["g"]
            cs[i] = self.exp[-1][i]["factor"]
            pt[i] = self.exp[-1][i]["power_t"]
            bs[i] = int(self.exp[-1][i]["branch"])
            
        return gs,cs,pt,bs
        
    def get_unique_gammas(self):
        '''Returns unique gammas and their counts.
        
        Returns
        -------
        g_un: numpy.ndarray of floats
            Unique gammas.
        n_g_un: int
            Count of unique gammas.
        '''
        g_un = np.unique(self.get_bare_params()[0])
        n_g_un = len(g_un)
        
        return g_un, n_g_un
        
    def get_branch_ampl(self,branch):
        '''Returns the amplitude of a branch.
        '''
        ampl = 0.0
        for step in self.steps:
            if step["branch"] == branch and\
               step["type"] in ["init","branch_path"]:
               ampl = step["A"]
               
        return ampl
        
    def get_integral(self):
        '''Return the integral from 0 to infinity of the EC.
        '''
        I = 0.0
        for ib in np.arange(self.n_branches):
            I += self.get_branch_ampl(ib)
            
        return I
                
    def get_branch_average_gamma(self,branch):
        '''Returns the average gamma of a branch.
        '''
        avg = 0.0
        n = 0
        for step in self.steps:
            if step["branch"] == branch:
                avg += step["g"]
                n +=1
                
        if n!=0: avg /= n
        return avg
        
    def get_ampl_ratio_fastest_slowest(self,return_all=False):
        '''Return the ratio between the amplitudes of the fastest and slowest
        branches.
        
        Returns
        -------
        ratio: float
            Ratio between the amplitude of the fastest branch (containing the 
            maximum gamma) and the one of the slowest branch (containing the 
            minimum gamma).
        
        '''
        if self.n_branches == 1: 
            if return_all: return None, None, None
            else: return None
        
        gs,cs,pt,bs = self.get_bare_params()
        argming = np.argmin(gs)
        argmaxg = np.argmax(gs)
        
        branchming = bs[argming]
        branchmaxg = bs[argmaxg]
        if branchming == branchmaxg: 
            #print("Fastest and slowest gammas in the same branch.")
            pass
        
        # Find the As of those two branches
        Amin = None
        Amax = None
        for step in self.steps:
            if step["type"] in ["init","branch_path"]:
                if step["branch"] == branchming:
                    Amin = step["A"]
                if step["branch"] == branchmaxg:
                    Amax = step["A"]
        
        return Amax/Amin, gs[argmaxg], gs[argming]
        
    def get_branches_ampl_avggamma(self):
        '''Return the amplitude and the average gamma for each branch.
        '''
        
        ampls = np.zeros(self.n_branches)
        gs = np.zeros(self.n_branches)
        for i in np.arange(self.n_branches):
            ampls[i] = self.get_branch_ampl(i)
            gs[i] = self.get_branch_average_gamma(i)
            
        return ampls,gs
        
    def has_gamma_faster_than(self,g_th):
        '''Returns whether this ExponentialConvolution contains a gamma
        faster (larger) than a given threshold.
        
        Parameters
        ----------
        g_th: float
            Threshold for gammas.
        
        Returns
        -------
        has_faster: bool
            Whether any gamma is larger than g_th.
        '''
        
        gs,cs,pt,bs = self.get_bare_params()
        has_faster = np.any(gs>g_th)
        
        return has_faster
        
    def get_ratio_to_peak(self,time):
        '''Returns the response normalized by its peak value.
        '''
        y = np.absolute(self.eval(time))
        
        return y/np.max(y)
        
    def get_peak_time(self,time,return_all=False):
        '''Get the peak time of the ExponentialConvolution (min or max, 
        depending of which has larger absolute value). 
        
        Keeping it simple and finding it through evaluation. Otherwise it would i
        nvolve solving a nonlinear equation and it would probably not be more 
        efficient. It therefore requires passing a time axis on which to look 
        for the peak time.
        
        Parameters
        ----------
        time: numpy.ndarray
            Time axis on which to evaluate the ExponentialConvolution.
        return_all: bool (optional)
            Whether to return also the index of the peak, and the evaluated
            array.
        
        Returns
        -------
        tstar: float
            Peak time.
        i: int
            Index of the peak time in the time array. Only returned if 
            return_all is True.
        y: numpy.ndarray
            The ExponentialConvolution evaluated on time. Only returned if 
            return_all is True.
        '''
        
        y = self.eval(time)
        mi = np.argmin(y)
        ma = np.argmax(y)
        
        tstar = time[mi] if -y[mi]>y[ma] else time[ma]
        if return_all:
            return tstar, mi if -y[mi]>y[ma] else ma, y
        else:
            return tstar
        
    def get_effective_decay_time(self,time,tmax=120.):
        '''Computes an effective decay time as the time between the peak and
        peak/e. Searches for the 1/e point between the peak time and tmax.
        
        Parameters
        ----------
        time: numpy.ndarray
            Time axis on which to evaluate the ExponentialConvolution and find
            the peak time. In the future could be replaced by maximisation of 
            the ExponentialConvolution.
        tmax: float (optional)
            Maximum possible absolute time of the 1/e point. If it's not within 
            this time, nan is returned. Default: 120.0.
            
        Returns
        -------
        tau: float
            Effective decay time of the ExponentialConvolution. np.nan is
            returned if the 1/e point is beyond tmax.
        '''
        
        tstar,istar,y = self.get_peak_time(time,return_all=True)
        peak_value = y[istar]
        
        # Check that the ExponentialConvolution decays before tmax
        ytmax = self.eval(tmax)
        if np.abs(ytmax)<np.abs(peak_value*np.exp(-1)):
            sol = root_scalar(self._eq_get_effective_tau,
                              method="brentq",bracket=[tstar,tmax],
                              args=peak_value)
            t2 = sol.root
        
            tau = t2-tstar
        else:
            tau = np.nan
            
        return tau
        
    def get_effective_rise_time(self,time):
        '''Computes an effective rise time as the time between peak/e and the 
        peak and. Searches for the 1/e point between 0 and the peak time.
        
        Parameters
        ----------
        time: numpy.ndarray
            Time axis on which to evaluate the ExponentialConvolution and find
            the peak time. In the future could be replaced by maximisation of 
            the ExponentialConvolution.
            
        Returns
        -------
        tau: float
            Effective decay time of the ExponentialConvolution. np.nan is
            returned if the peak time is the last element of time.
        '''
        
        tstar,istar,y = self.get_peak_time(time,return_all=True)
        if tstar == 0: return 0.0
        peak_value = y[istar]
        
        # Check that the peak is not at the end of the time axis (i.e. the 
        # time axis is too short to do this thing.
        if tstar<time[-1]:
            sol = root_scalar(self._eq_get_effective_tau,
                              method="brentq",bracket=[0,tstar],
                              args=peak_value)
            t2 = sol.root
        
            tau = tstar-t2
        else:
            tau = np.nan
            
        return tau
        
        
    def _eq_get_effective_tau(self,t,peak_value):
        '''Equation of which to find the root.
        '''
        y = self.eval(t)
        return y-peak_value*np.exp(-1)
        


    ###################################################
    # METHODS TO RETURN PARTIAL ExponentialConvolutions
    ###################################################
    
    def drop_branches(self,branches,mode="discard"):
        '''Returns a new instance of ExponentialConvolution discarding (or 
        retaining only) a given set of branches. Useful to test significance of 
        branches.
        
        Parameters
        ----------
        branches: int or array_like of int
            Branches to be discarded or retained.
        mode: str (optional)
            If mode is "discard", the listed branches will be discarded. If mode
            is "retain", the listed branches will be retained.
            
        Returns
        -------
        new_ec: ExponentialConvolution
            New ExponentialConvolution object.
        
        '''
        try: len(branches)
        except: branches = [branches]
        
        new_steps = []
        for i in np.arange(len(self.steps)):
            if self.steps[i]["type"] != "init":
                cur_b = self.steps[i]["branch"]
            else:
                cur_b = 0
            if (mode=="discard" and cur_b not in branches) or\
               (mode=="retain" and cur_b in branches):
                new_steps.append(self.steps[i])
        
        if len(new_steps)==0: print(self.steps)
        if new_steps[0]["type"] == "branch_path":
            new_steps[0]["type"] = "init"
        
        new_ec = self.from_steps(new_steps)
        return new_ec
        
    def drop_gammas(self,gammas,mode="discard"):
        '''Returns a new instance of ExponentialConvolution discarding (or 
        retaining only) a given set of gammas. Useful to test significance of 
        gammas.
        
        Parameters
        ----------
        gammas: int or array_like of int
            Gammas to be discarded or retained.
        mode: str (optional)
            If mode is "discard", the listed gammas will be discarded. If mode
            is "retain", the listed gammas will be retained.
            
        Returns
        -------
        new_ec: ExponentialConvolution
            New ExponentialConvolution object.
        
        '''
        try: len(gammas)
        except: gammas = [gammas]
        
        new_steps = []
        for i in np.arange(len(self.steps)):   
            if (mode=="discard" and self.steps["g"] not in gammas) or\
               (mode=="retain" and self.steps["g"] in gammas):
                new_steps.append(self.steps[i])
                
        new_ec = self.from_steps(new_steps)
        return new_ec
    
    def drop_gammas_larger_than(self,gamma_th):
        '''Returns a new instance of ExponentialConvolution discarding gammas
        above a threshold. Useful to test significance of gammas in comparisons
        of multiple ExponentialConvolutions.
        
        Parameters
        ----------
        gamma_th: float
            Threshold over which to discard gammas (strictly larger than).
            
        Returns
        -------
        new_ec: ExponentialConvolution
            New ExponentialConvolution object.
        
        '''
        
        if np.isinf(gamma_th):
            return self.from_steps(self.steps)
        
        new_steps = []
        for i in np.arange(len(self.steps)):   
            if self.steps[i]["g"]<=gamma_th:
                new_steps.append(self.steps[i])
        
        if len(new_steps)==0:
            return None
        else:
            if new_steps[0]["type"] == "convolve":
                b = new_steps[0]["branch"]
                if b==0: 
                    A = self.steps[0]["A"]
                else:
                    steps_b = [s for s in self.steps[1:] if s["branch"]==b]
                    
                    for sb in steps_b:
                        if sb["type"] in ["init","branch_path"]:
                            A = sb["A"]
                new_steps[0]["type"] = "init"
                new_steps[0]["A"] = A
            elif new_steps[0]["type"] == "branch_path":
                new_steps[0]["type"] = "init"

            new_ec = self.from_steps(new_steps)
            return new_ec
    
    def find_unresolvable_gammas(self,time,atol=1e-3):
        '''Finds the threshold for gamma above which convolving 
        exp(-gamma*time) with this ExponentialConvolution produces an average 
        absolute error smaller than rtol.
        
        Parameters
        ----------
        time: array_like of floats
            Time axis on which to compare the convolution with the original
            ExponentialConvolution.
        atol: float (optional)
            Maximum absolute error. Default: 1e-3.
        
        Returns
        -------
        g_max: float
            Threshold gamma.
        
        '''
        b0 = 0.3
        b1 = 10.0
        s0 = np.sign(self.error_by_additional_convolution(b0,time,atol))
        s1 = np.sign(self.error_by_additional_convolution(b1,time,atol))
        
        if s1==s0: 
            return np.inf
            #raise ValueError("The brackets in "+\
            #    "ExponentialConvolution.find_insignificant_gammas() are wrong.")
        
        g_max = root_scalar(self.error_by_additional_convolution,
                          args=(time,atol),bracket=[b0,b1],
                          method="brentq").root
        
        return g_max
        
    def error_by_additional_convolution(self,gamma,time,atol):
        '''Equation of which to find the root for find_unresolvable_gammas().
        '''
        a = self.eval(time)
        b = gamma*np.exp(-gamma*time)
        if np.all(b[1:]<gamma*1e-2): return -10.
        ap = pp.convolution(a, b, time[1]-time[0], 8)
        err = np.average(np.abs(ap-a))
        
        return err-gamma*atol
        
    def drop_saturation_branches(self):
        # Get the derivative at time 0+ to determine if you need to drop 
        # negative or positive branches.
        d = self.get_derivative2_t_0()
        
        # Find the positive branches, which you will discard or retain depending
        # on the sign of d.
        pbranches = []
        As = []
        for ib in np.arange(self.n_branches):
            if ib==0:
                A = self.steps[0]["A"]
            else:
                A = [st["A"] for st in self.steps if (st["type"]=="branch_path" and st["branch"]==ib)]
                if len(A)>1: print("NONONONO")
                A = A[0]
            if A>0: pbranches.append(ib)
            As.append(A)
            
        if d>0: mode = "retain"
        elif d<0: mode = "discard"
            
        new_ec = self.drop_branches(pbranches,mode=mode)
        
        return new_ec
        
        
    ################################
    # CONVOLUTIONAL GAMMA CLUSTERING
    ################################
        
    def cluster_gammas(self,g0,g1):
        print("Still needs implementation for t^n e^(-gt)")
        gs = [e["g"] for e in self.exp[-1]]
        if not (g0 in gs and g1 in gs): return None
        self.exp[-1] = [e for e in self.exp[-1] if e["g"] not in [g0,g1]]
        
        for i in np.arange(len(self.exp[-1])):
            gi = self.exp[-1][i]["g"]
            self.exp[-1][i]["factor"] *= (g0-gi)/g0 * (g1-gi)/g1
            
        gnew = 0.5*(g0+g1)
        self.convolve_exp(gnew)
        self.convolve_exp(gnew)
    
    @classmethod
    def _cluster_gammas_convolutional_1(cls,ec):
        # Find the closest gammas in each branch and determine which pair
        # should be clustered.
        
        qs = []
        rs = []
        min_d = []
        gss = []
        As = []
        no_more_mismatch = True
        for ib in np.arange(ec.n_branches):
            gs = []
            for step in ec.steps:
                in_this_branch = False
                if step["type"]=="init": in_this_branch = ib==0
                elif step["branch"]==ib: in_this_branch = True
                if in_this_branch:
                    gs.append(step["g"])
                    if step["type"] in ["init","branch_path"]:
                        As.append(step["A"])
            gs = np.array(gs)
            if len(np.unique(gs))!=1: no_more_mismatch = False
            gss.append(gs)
            
            d = np.square(gs[:,None]-gs[None,:])
            d[np.diag_indices(len(d))] = 1e6
            d[d==0] = 1e6
            
            q = np.argmin(d,axis=1)
            q_d = np.min(d,axis=1)
            r = np.argmin(q_d)
            q = q[r]
            
            qs.append(q)
            rs.append(r)
            min_d.append(np.abs(d[q,r]))
        
        # Find the pair of gammas that are the closest.
        sb = np.argmin(min_d) # Branch with the gammas to be clustered
        q = qs[sb] # One of the two gammas.
        r = rs[sb] # The other.
        # In addition to these, you need to see if there is any other gamma
        # that is already equal to either of the two, because these other 
        # gammas also need to be substituted with the new (average) clustered 
        # gamma.
        rprime = np.where(gss[sb]==gss[sb][r])
        qprime = np.where(gss[sb]==gss[sb][q])
        n_r = len(rprime[0])
        n_q = len(qprime[0])
        
        # Calculate the new gamma as the weighted average of the old gammas
        # and substitute the old gammas
        g_new = (n_q*gss[sb][q]+n_r*gss[sb][r])/(n_r+n_q)
        gss[sb][r] = g_new
        gss[sb][q] = g_new
        gss[sb][rprime] = g_new
        gss[sb][qprime] = g_new
        
        steps_new = []
        for ib in np.arange(ec.n_branches):
            for i_g in np.arange(len(gss[ib])):
                if i_g==0 and ib==0:
                    dic = {"type":"init",  "g": gss[ib][i_g],
                           "A": As[ib], "branch":0}
                elif i_g==0 and ib>0:
                    dic = {"type":"branch_path",  "g": gss[ib][i_g],
                           "A": As[ib],  "branch": ib}
                elif i_g>0:
                    dic = {"type":"convolve", "g": gss[ib][i_g],
                           "branch": ib}
                steps_new.append(dic)
        
        ec_new = cls.from_steps(steps_new)
        return ec_new, no_more_mismatch
        
    @classmethod
    def _cluster_gammas_convolutional_across_branches_1(cls,ec):
        gs,cs,pt = cls.get_bare_params()
        
        return None
        
    @classmethod
    def cluster_gammas_convolutional(cls,n,ec,x=None,atol=1e-5,across_branches=False):
        if x is not None: 
            y_old = ec.eval(x)
        
        i = 0
        while True:
            ec_tmp, no_more_mismatch = cls._cluster_gammas_convolutional_1(ec)
            if no_more_mismatch: ec = ec_tmp; break
            if x is None:
                if i==n: ec = ec_tmp; break
            else:
                y = ec_tmp.eval(x)
                if not np.all(np.abs(y-y_old)<atol): break
            ec = ec_tmp
            i += 1
            
        #if across_branches:
        #    ec.
        
        return ec
        
    ###########################
    # ADDITIVE GAMMA CLUSTERING
    ###########################
        
    @classmethod
    def _cluster_gammas_additive_1(cls,gs,cs):
        '''Using the additive method, cluster the two closest gammas.
        '''
        # Find two closest gammas
        d = np.square(gs[:,None]-gs[None,:])
        d[np.diag_indices(len(d))] = 1e6
        
        q = np.argmin(d,axis=1)
        q_d = np.min(d,axis=1)
        r = np.argmin(q_d)
        q = q[r]
        
        gs_old = np.array([gs[q],gs[r]])
        cs_old = np.array([cs[q],cs[r]])
        
        g_new, c_new = cls.equivalence_1_many(gs_old,cs_old)
        
        gs_new = np.delete(gs,(q,r))
        cs_new = np.delete(cs,(q,r))
        
        gs_new = np.append(gs_new,g_new)
        cs_new = np.append(cs_new,c_new)
        
        return gs_new,cs_new
    
    @classmethod
    def cluster_gammas_additive(cls,gs,cs,n):
        '''Using the additive method, cluster the n closest gammas.
        '''
        for i in np.arange(n):
            gs,cs = cls._cluster_gammas_additive_1(gs,cs)
        
        return gs,cs
        
    @classmethod
    def cluster_gammas_additive_auto(cls,x,gs,cs,eps=10.):
        '''Automatically decide how many gammas to cluster with the additive
        method.
        '''
        pt = np.zeros_like(gs)
        y_old = cls.eval2(x,gs,cs,pt)
        
        e = 0.
        j = 0
        while True:
            gs_new,cs_new = cls._cluster_gammas_additive_1(gs,cs)
            y_new = cls.eval2(x,gs_new,cs_new,pt)
            msk = y_old!=0
            e = np.sum(np.power((y_old[msk]-y_new[msk])/y_old[msk],2))/len(x)
            if e>eps:break
            gs = gs_new
            cs = cs_new
            j+=1
            
        return gs,cs,j
        
               
    @staticmethod
    def _equivalence_1_many(gs,cs,g,c):
        y = c/(2*g**2)
        y+= -2*np.sum(cs/(gs+g)**2)
        
        return y
        
    @classmethod
    def equivalence_1_many(cls,gs,cs):
        c = np.sum(cs)
        f = lambda g: cls._equivalence_1_many(gs,cs,g,c)
        
        sol = root(f,np.average(gs))
        
        return sol.x,c
        
    ##########################
    # ACROSS-OBJECT CLUSTERING
    ##########################
    @classmethod
    def _acrob_cluster_1(cls,objs):
        # Find the two gammas that are the closest across the 
        # ExponentialConvolution objects.
        gss = []
        for obj in objs:
            gs,_,_,_ = obj.get_bare_params()
            gss.append(gs)
        
        min_ds = []
        argmins = []
        for i in np.arange(len(objs)):
            min_ds_tmp = []
            argmins_tmp = []
            # For each i-th object iterate over all the j-th objects, and find
            # the k-th and l-th gammas that are the closest ones.
            for j in np.arange(len(objs)):
                if i==j: 
                    argmins_tmp.append({"i":None})
                    min_ds_tmp.append(1e6)
                else:
                    dist = np.abs(gss[i][:,None]-gss[j][None,:])
                    dist[dist==0] = 1e6
                    argmin = np.argmin(dist)
                    k = argmin//len(gss[j])
                    l = argmin%len(gss[j])
                    min_d = np.abs(gss[i][k]-gss[j][l])
                    min_ds_tmp.append(min_d)
                    argmins_tmp.append({"i":i,"j":j,"k":k,"l":l})
            # For the i-th object, find the closest gamma among all the j-th
            # objects.
            j_star = np.argmin(min_ds_tmp)
            argmins.append(argmins_tmp[j_star])
            min_ds.append(min_ds_tmp[j_star])
        
        # Now find the closest pair of gammas throughout the whole set of 
        # objects.
        i_star = np.argmin(min_ds)
        argmin_star = argmins[i_star]
        # Why did I put this assertion in the first place? This throws an
        # assertion error in some cases, probably because argmin_star["i"] is 
        # None. I'm just going to allow also None.
        assert i_star==argmin_star["i"] or argmin_star["i"] is None
        
        if argmin_star["i"] is not None:
            i_star = argmin_star["i"] # Index of the i-th object
            j_star = argmin_star["j"] # Index of the j-th object
            k_star = argmin_star["k"] # Index of the gamma of the i-th object
            l_star = argmin_star["l"] # Index of the gamma of the i-th object
            # Take the average of the gammas - TODO If you want to do a weighted
            # average, take only one count per branch (i.e. with t^2 e^igamma t, 
            # gamma should be counted only once, not three times).
            gamma_new = 0.5*(gss[i_star][k_star]+gss[j_star][l_star])
            
            # Replace the gamma to be clustered with gamma_new and make the new
            # ExponentialConvolution objects.
            
            objs_out = []
            for ii in np.arange(len(objs)):
                new_steps = []
                for s in objs[ii].steps:
                    new_s = s.copy()
                    if new_s["g"] in [gss[i_star][k_star],gss[j_star][l_star]]: 
                        new_s["g"] = gamma_new
                    new_steps.append(new_s)
                objs_out.append(cls.from_steps(new_steps))
            
            # Do you need to rescale also the coefficients? By something
            # like (new_g/old_g)^(pt+1)? No, because you're re-building the 
            # object with the steps, so the rescaling of the coefficient is done
            # automatically.
        else:
            objs_out = []
            for ii in np.arange(len(objs)):
                objs_out.append(cls.from_steps(objs[ii].steps))
        
        return objs_out
        
            
    @classmethod
    def acrob_cluster(cls,n,objs,x=None,atol=1e-5):
        ecs = objs.copy()
        if x is not None: 
            assert np.all(x>=0.0)
            y_old = np.zeros((len(objs),len(x)))
            y_new = np.zeros((len(objs),len(x)))
        
        i = 0
        while True:
            if x is not None:
                for i_ec in np.arange(len(objs)): 
                    y_old[i_ec] = ecs[i_ec].eval(x)
            
            ecs = cls._acrob_cluster_1(ecs)
            i += 1            
            if i==n: break
            if x is not None:
                have_to_break = False
                for i_ec in np.arange(len(objs)): 
                    y_new[i_ec] = ecs[i_ec].eval(x)
                    have_to_break = have_to_break or np.any(np.abs(y_old[i_ec]-y_new[i_ec])>atol)
                    
                if have_to_break: break
            
        return ecs
        
    @classmethod
    def find_matching_gammas(cls,objs,rtol=1e-3,rtol_tau=None,
                             discard_unresolvable=True,time=None,
                             atol_unresolvable=None,return_all=False):
        '''Finds matching gammas across a list of ExponentialConvolution 
        objects, and computes the fraction and ordered fraction matrices,
        which tell you what fraction of the gammas of the j-th 
        ExponentialConvolution is contained in the i-th one.
        If discard_unresolvable is True, in computing the number of unique gammas,
        used as denominator in the fraction matrices, the function will discard
        from the supposedly upstream neuron insignificant gammas which, even
        if convolved with the downstream neuron, would not change the kernel
        a lot, and therefore can be lost by the fit.
        
        Parameters
        ----------
        objs: list of ExponentialConvolution
            Objects to compare.
        rtol: float (optional)
            Relative tolerance in the comparison of the gammas. Default: 1e-3
        rtol_tau: float (optional)
            Relative tolerance in the comparison of the taus. If not None, the
            comparison will be made on the taus and not the gammas. 
            Default: None.
        discard_unresolvable: bool (optional)
            Whether to discard insignificant gammas in the comparison, using
            ExponentialConvolution.find_unresolvable_gammas() and
            ExponentialConvolution.drop_gammas_larger_than(). Default: True.
        time: array_like of floats
            Time axis. Required if discard_unresolvable is True. Default: None.
        atol_unresolvable: float (optional)
            Absolute tolerance to be passed to discard_unresolvablenificant_gammas.
            If None, it is set equal to rtol.
        return_all: bool (optional)
            Whether to return also fraction and fraction_ordered.
            
        Returns
        -------
        matches: list of lists of dictionaries
            matches[i_ec][j] is a dictionary describing a match between
            the g_orig gamma in the i_ec-th object with a gamma from another
            object. The dictionary's keys are "i_ec" (index of the other 
            object), "g" (gamma of the other object), "b" (branch in the other
            object), "g_orig" (gamma in the i_ec-th object), 
            "b_orig" (branch in the i_ec-th object).
        '''
        # If rtol_tau is not None look for matching taus TODO
        
        if discard_unresolvable:
            if atol_unresolvable is None: atol_unresolvable=rtol
            if time is None:
                raise ValueError("You need to pass time to "+\
                        "ExponentialConvolutionfind_matching_gammas() if"+\
                        "discard_unresolvable is True.")
        
        # Get all the bare parameters
        gss = []
        bss = []
        i_ecs = []
        for i_ec in np.arange(len(objs)):
            gs,cs,pt,bs = objs[i_ec].get_bare_params()
            for i_p in np.arange(len(gs)): 
                gss.append(gs[i_p]) 
                bss.append(bs[i_p])
                i_ecs.append(i_ec)
        
        # Build the matches dictionaries
        matches = [[] for i_ec in np.arange(len(objs))]
        for i_p in np.arange(len(gss)):
            matching_g_is = np.where( (np.abs(gss-gss[i_p])<rtol*gss[i_p] ) * (i_ecs!=i_ecs[i_p]) )[0]
            for m in matching_g_is:
                a = {"i_ec": i_ecs[m], 
                     "g":gss[m],
                     "b":bss[m],
                     "g_orig":gss[i_p],
                     "b_orig":bss[i_p]}
                matches[i_ecs[i_p]].append(a)
                
        if not return_all:
            return matches
                
        # (Probably not the most efficient sequence of for loops, but ok)
        # For each object, compute the ratio between the gammas from other
        # objects it contains, and the total number of gammas that other object
        # has. (Does the object i contain 100% of the object j's gammas?)
        fraction = np.zeros((len(objs),len(objs)))
        for i_ec in np.arange(len(objs)):
            if discard_unresolvable:
                # Discard from j the insignificant gammas that would be lost if 
                # convolved with i (and therefore should not be counted because
                # the fit likely does not see them). First, calculate the
                # minimum gamma that is not seen if convolved in i. You'll
                # discard the gammas from j in the for loop below.
                g_min = objs[i_ec].find_unresolvable_gammas(time,atol_unresolvable)
                
            for j_ec in np.arange(len(objs)):
                
                if discard_unresolvable:
                    tmp_ec_j = objs[j_ec].drop_gammas_larger_than(g_min)
                    if tmp_ec_j is None: 
                        # No gamma remaining after discarding insignificant gammas.
                        tmp_ec_j = objs[j_ec]
                else:
                    tmp_ec_j = objs[j_ec]
                
                n_g_unique_in_j = tmp_ec_j.get_unique_gammas()[1]
                n_g_i_matched_to_j_unique = len(np.unique([ m["g"] for m in matches[i_ec] if m["i_ec"]==j_ec]))
                fraction[i_ec,j_ec] = n_g_i_matched_to_j_unique/n_g_unique_in_j
        
        # Make an ordered version of the fraction matrix. Keep non-zero entries
        # only if i contains a greater fraction of j's gammas than j does of 
        # i's. So you should keep the entry only if fraction[i,j] is greater
        # than fraction[j,i].
        fraction_ordered = np.zeros((len(objs),len(objs)))
        for i_ec in np.arange(len(objs)):
            for j_ec in np.arange(len(objs)):
                if fraction[i_ec,j_ec] >= fraction[j_ec,i_ec]:
                    fraction_ordered[i_ec,j_ec] = fraction[i_ec,j_ec]
                    
                    
        return matches, fraction, fraction_ordered
            
            
    
    
        
