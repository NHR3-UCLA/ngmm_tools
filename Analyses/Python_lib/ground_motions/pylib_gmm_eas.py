# ba18.py
# Conversion of Jeff Bayless' MATLAB code to Python
# Including class ba18
# I've tried to avoid mixed UPPER and lower case variable names
#   e.g. Mbreak, Rrup, Vsref 

#arithmetic libraries
import numpy as np
import numpy.matlib
from scipy import linalg as scipylalg
from scipy import sparse as scipysp
#geographic coordinates
import pyproj
#statistics libraries
import pandas as pd
#geometric libraries
from shapely.geometry import Point as shp_pt, Polygon as shp_poly

def SlicingSparceMat(mat_sp, i_rows, j_col):
    '''Slice sparse matrix'''

    return np.array([mat_sp.getcol(i_r).toarray().flatten()[j_col] for i_r in i_rows])  

def QuartCos(per, x0, x, flag_left = False):
    
    y = np.cos( 2.*np.pi*(x-x0)/per )
    if flag_left: y[np.logical_or(x < x0-per/4, x > x0)]     = 0.
    else:         y[np.logical_or(x < x0,       x > x0+per/4)] = 0.
    
    return y

def QuadCosTapper(freq, freq_nerg):
    
    #boxcar at intermediate frequencies
    i_box = np.logical_and(freq >= freq_nerg.min(), freq <= freq_nerg.max()) 
    y_box        = np.zeros(len(freq))
    y_box[i_box] = 1.
    #quarter cosine left taper
    per   = 2 * freq_nerg.min()
    y_tpl = QuartCos(per, freq_nerg.min(), freq, flag_left=True)
    #quarter cosine right taper
    per   = 2 * freq_nerg.max()
    y_tpr = QuartCos(per, freq_nerg.max(), freq)
    #combined tapering function
    y_tapper = np.array([y_box, y_tpl, y_tpr]).max(axis=0)

    return y_tapper

def TriagTapper(freq, freq_nerg):
    
    fn_min = freq_nerg.min()
    fn_max = freq_nerg.max()
    
    #triangular window
    f_win = np.array([0.5*fn_min, fn_min, fn_max, 1.5*fn_max])
    y_win = np.array([0.,          1.,     1.,     0.])
    
    #triangular tapering function
    y_tapper = np.interp(np.log(freq), np.log(f_win), y_win)

    return y_tapper

def ConvertPandasDf2NpArray(df_array):
    
    array = df_array.values if isinstance(df_array, pd.DataFrame) or isinstance(df_array, pd.Series) else df_array
    
    return array

    

class BA18:
    def __init__(self, file=None):
        '''
        Constructor for this class
        Read CSV file of BA18 coefficients, frequency range: 0.1 - 100 Hz

        Parameters
        ----------
        file : string, optional
            file name for coefficients. The default is None.
        '''

        if file is None:
                file = '/mnt/halcloud_nfs/glavrent/Research/Nonerg_CA_GMM/Analyses/Python_lib/ground_motions/Bayless_ModelCoefs.csv'
        df = pd.read_csv(file, index_col=0)
        df = df.head(301)
        # Frequencies 0.1 - 24 Hz
        self.freq = df.index.values
        # Median FAS parameters
        self.b1 = df.c1.values
        self.b2 = df.c2.values
        self.b3quantity = df['(c2-c3)/cn'].values
        self.b3 = df.c3.values
        self.bn = df.cn.values
        self.bm = df.cM .values
        self.b4 = df.c4.values
        self.b5 = df.c5.values
        self.b6 = df.c6.values
        self.bhm = df.chm.values
        self.b7 = df.c7.values
        self.b8 = df.c8.values
        self.b9 = df.c9.values
        self.b10 = df.c10.values
        self.b11a = df.c11a.values 
        self.b11b = df.c11b.values
        self.b11c = df.c11c.values
        self.b11d = df.c11d.values
        self.b1a = df.c1a.values
        self.b1a[239:] = 0
        # Non-linear site parameters
        self.f3 = df.f3.values
        self.f4 = df.f4.values
        self.f5 = df.f5.values
        # Aleatory variability parameters
        self.s1 = df.s1.values
        self.s2 = df.s2.values
        self.s3 = df.s3.values
        self.s4 = df.s4.values
        self.s5 = df.s5.values
        self.s6 = df.s6.values
        # Constants
        self.b4a = -0.5
        self.vsref = 1000
        self.mbreak = 6.0
        #bedrock anelastic attenuation
        self.b7rock = self.b7.copy()
        #frequency limits
        # self.maxfreq = 23.988321
        self.maxfreq = self.freq.max()
        self.minfreq = self.freq.min()
    
    def EasBase(self, mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7 = True):
                        
        # note Z1 must be provided in km
        z1ref = (1/1000) * np.exp(-7.67/4 * np.log((vs30**4+610**4)/(1360**4+610**4)) )
        if vs30<=200:
            self.b11 = self.b11a
        if vs30>200 and vs30<=300:
            self.b11 = self.b11b
        if vs30>300 and vs30<=500:
            self.b11 = self.b11c
        if vs30>500:
            self.b11 = self.b11d
    
        if z1 is None or np.isnan(z1):
            z1 = self.Z1(vs30, regid=1)
    
        # Compute lnFAS by summing contributions, including linear site response
        lnfas = self.b1 + self.b2*(mag-self.mbreak)
        lnfas += self.b3quantity*np.log(1+np.exp(self.bn*(self.bm-mag))) 
        lnfas += self.b4*np.log(rrup+self.b5*np.cosh(self.b6*np.maximum(mag-self.bhm,0)))
        lnfas += (self.b4a-self.b4) * np.log( np.sqrt(rrup**2+50**2) ) 
        lnfas += self.b7 * rrup if flag_keep_b7 else 0.
        lnfas += self.b8 * np.log( min(vs30,1000) / self.vsref ) 
        lnfas += self.b9 * min(ztor,20) 
        lnfas += self.b10 * fnorm 
        lnfas += self.b11 * np.log( (min(z1,2) + 0.01) / (z1ref + 0.01) )
        # this is the linear spectrum up to maxfreq=23.988321 Hz
        maxfreq = 23.988321
        imax = np.where(self.freq==maxfreq)[0][0]
        fas_lin = np.exp(lnfas)
        # Extrapolate to 100 Hz
        fas_maxfreq = fas_lin[imax]
        # Kappa
        kappa = np.exp(-0.4*np.log(vs30/760)-3.5)
        # Diminuition operator
        D = np.exp(-np.pi*kappa*(self.freq[imax:] - maxfreq))
        fas_lin = np.append(fas_lin[:imax], fas_maxfreq * D)
        
        # Compute non-linear site response
        # get the EAS_rock at 5 Hz (no c8, c11 terms)
        vref=760
        #row = df.iloc[df.index == 5.011872]
        i5 = np.where(self.freq==5.011872)
        lnfasrock5Hz = self.b1[i5]
        lnfasrock5Hz += self.b2[i5]*(mag-self.mbreak) 
        lnfasrock5Hz += self.b3quantity[i5]*np.log(1+np.exp(self.bn[i5]*(self.bm[i5]-mag))) 
        lnfasrock5Hz += self.b4[i5]*np.log(rrup+self.b5[i5]*np.cosh(self.b6[i5]*max(mag-self.bhm[i5],0)))
        lnfasrock5Hz += (self.b4a-self.b4[i5])*np.log(np.sqrt(rrup**2+50**2)) 
        lnfasrock5Hz += self.b7rock[i5]*rrup 
        lnfasrock5Hz += self.b9[i5]*min(ztor,20) 
        lnfasrock5Hz += self.b10[i5]*fnorm
        # Compute PGA_rock extimate from 5 Hz FAS
        IR = np.exp(1.238+0.846*lnfasrock5Hz)
        # apply the modified Hashash model
        self.f2 = self.f4*( np.exp(self.f5*(min(vs30,vref)-360)) - np.exp(self.f5*(vref-360)) )
        fnl0 = self.f2 * np.log((IR+self.f3)/self.f3)
        fnl0[np.where(fnl0==min(fnl0))[0][0]:] = min(fnl0)
        fas_nlin = np.exp( np.log(fas_lin) + fnl0 )
        
        # Aleatory variability
        if mag<4:
            tau = self.s1
            phi_s2s = self.s3
            phi_ss = self.s5
        if mag>6:
            tau = self.s2
            phi_s2s = self.s4
            phi_ss = self.s6
        if mag >= 4 and mag <= 6:
            tau = self.s1 + ((self.s2-self.s1)/2)*(mag-4)
            phi_s2s = self.s3 + ((self.s4-self.s3)/2)*(mag-4)
            phi_ss = self.s5 + ((self.s6-self.s5)/2)*(mag-4)
        sigma = np.sqrt(tau**2 + phi_s2s**2 + phi_ss**2 + self.b1a**2);
        
        return self.freq, fas_nlin, fas_lin, sigma

    def EasBaseArray(self, mag, rrup, vs30, ztor, fnorm, z1=None, regid=1, flag_keep_b7=True):
        
        #convert eq parameters to np.arrays   
        mag   = np.array([mag]).flatten()
        rrup  = np.array([rrup]).flatten()
        vs30  = np.array([vs30]).flatten()
        ztor  = np.array([ztor]).flatten()
        fnorm = np.array([fnorm]).flatten()
        z1    = np.array([self.Z1(vs, regid) for vs in vs30]) if z1 is None else np.array([z1]).flatten()
        
        #number of scenarios
        npt = len(mag)
        #input assertions
        assert( np.all(npt == np.array([len(rrup),len(vs30),len(ztor),len(fnorm),len(z1)])) ),'Error. Inconsistent number of gmm parameters'
        
        #compute fas for all scenarios
        fas_nlin = list()
        fas_lin  = list()
        sigma    = list()
        for k, (m, r, vs, zt, fn, z_1) in enumerate(zip(mag, rrup, vs30, ztor, fnorm, z1)):
            ba18_base = self.EasBase(m, r, vs, zt, fn, z_1, regid, flag_keep_b7)[1:]
            fas_nlin.append(ba18_base[0])
            fas_lin.append(ba18_base[1])
            sigma.append(ba18_base[2])
        #combine them to np.arrays
        fas_nlin = np.vstack(fas_nlin)
        fas_lin  = np.vstack(fas_lin)
        sigma    = np.vstack(sigma)
        
        # if npt == 1 and flag_flatten:
        #    fas_nlin = fas_nlin.flatten()
        #    fas_lin  = fas_lin.flatten()
        #    sigma    = sigma.flatten() 
        
        #return self.EasBase(mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7)
        return self.freq, fas_nlin, fas_lin, sigma    

    def Eas(self, mag, rrup, vs30, ztor, fnorm, z1=None, regid=1, flag_keep_b7=True, flag_flatten=True):
        '''
        Computes BA18 EAS GMM for all frequencies

        Parameters
        ----------
        mag : real
            moment magnitude [3-8].
        rrup : real
            Rupture distance in kilometers (km) [0-300].
        vs30 : real
            site-specific Vs30 = slowness-averaged shear wavespeed of upper 30 m (m/s) [120-1500].
        ztor : real
            depth to top of rupture (km) [0-20].
        fnorm : real
            1 for normal faults and 0 for all other faulting types (no units) [0 or 1].
        z1 : real, optional
            site-specific depth to shear wavespeed of 1 km/s (km) [0-2]. The default is =None.
        regid : int, optional
            DESCRIPTION. The default is =1.

        Returns
        -------
        freq : np.array
            frequency array.
        fas_nlin : np.array
            fas array with nonlinear site response.
        fas_lin : np.array
            fas array with linear site response.
        sigma : np.array
            standard deviation array.
        '''
        
        #return self.EasBase(mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7)
        # return self.EasBaseArray(mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7, flag_flatten)
        
        freq, fas_nlin, fas_lin, sigma = self.EasBaseArray(mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7)
        
        #flatten arrays if only one datapoint
        if fas_nlin.shape[0] == 1 and flag_flatten:
            fas_nlin = fas_nlin.flatten()
            fas_lin  = fas_lin.flatten()
            sigma    = sigma.flatten()         
        
        return freq, fas_nlin, fas_lin, sigma 

    
    def EasF(self, freq, mag, rrup, vs30, ztor, fnorm, z1=None, regid=1, flag_keep_b7 = True, flag_flatten=True):
        '''
        Computes BA18 EAS GMM for frequency of interest

        Parameters
        ----------
        mag : real
            moment magnitude [3-8].
        rrup : real
            Rupture distance in kilometers (km) [0-300].
        vs30 : real
            site-specific Vs30 = slowness-averaged shear wavespeed of upper 30 m (m/s) [120-1500].
        ztor : real
            depth to top of rupture (km) [0-20].
        fnorm : real
            1 for normal faults and 0 for all other faulting types (no units) [0 or 1].
        z1 : real, optional
            site-specific depth to shear wavespeed of 1 km/s (km) [0-2]. The default is =None.
        regid : int, optional
            DESCRIPTION. The default is =1.

        Returns
        -------
        freq : real
            frequency of interest.
        fas_nlin : real
            fas with nonlinear site response for frequency of interest.
        fas_lin : real
            fas with linear site response for frequency of interest.
        sigma : real
            standard deviation of frequency of interest.
        '''
        
        #convert freq to numpy array
        freq = np.array([freq]).flatten()
        
        #frequency tolerance
        f_tol = 1e-4
        #compute fas for all frequencies
        freq_all, fas_all, fas_lin_all, sig_all = self.EasBaseArray(mag, rrup, vs30, ztor, fnorm, z1, regid, flag_keep_b7)
        
        #find eas for frequency of interest
        if np.all([np.isclose(f, freq_all, f_tol).any() for f in freq]):
            # i_f     = np.array([np.where(np.isclose(f, freq_all, f_tol))[0] for f in freq]).flatten()
            i_f     = np.array([np.argmin(np.abs(f-freq_all)) for f in freq]).flatten()
            freq    = freq_all[i_f]
            fas     = fas_all[:,i_f]
            fas_lin = fas_lin_all[:,i_f]
            sigma   = sig_all[:,i_f]
        else:
            fas     = np.vstack([np.exp(np.interp(np.log(np.abs(freq)), np.log(freq_all), np.log(fas),   left=-np.nan, right=-np.nan)) for fas   in fas_all])
            fas_lin = np.vstack([np.exp(np.interp(np.log(np.abs(freq)), np.log(freq_all), np.log(fas_l), left=-np.nan, right=-np.nan)) for fas_l in fas_lin_all])
            sigma   = np.vstack([       np.interp(np.log(np.abs(freq)), np.log(freq_all), sig,           left=-np.nan, right=-np.nan)  for sig   in sig_all])

        #if one scenario flatten arrays        
        if fas.shape[0] == 1 and flag_flatten:
            fas     = fas.flatten()
            fas_lin = fas_lin.flatten()
            sigma   = sigma.flatten()

        return fas, fas_lin, sigma
    
    def GetFreq(self):
        
        return np.array(self.freq)
    
    def Z1(self, vs30, regid=1):
        '''
        Compute Z1.0 based on Vs30 for CA and JP

        Parameters
        ----------
        vs30 : real
            Time average shear-wave velocity.
        regid : int, optional
            Region ID. The default is 1.

        Returns
        -------
        real
            Depth to a shear wave velocity of 1000m/sec.
        '''
        
        if regid == 1:    #CA
            z_1 = -7.67/4. * np.log((vs30**4+610.**4)/(1360.**4+610.**4))        
        elif regid == 10: #JP
            z_1 = -5.23/4. * np.log((vs30**4+412.**4)/(1360.**4+412.**4))
    
        return 1/1000*np.exp(z_1)



