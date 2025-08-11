#  DEMO script (python version) for pyseisdl
#  
#  Copyright (C) 2022 Yangkang Chen
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details: http://www.gnu.org/licenses/
#  
## generate synthetic data
#This synthetic data was used in Huang et al., 2016, Damped multichannel singular spectrum analysis for 3D random noise attenuation, Geophysics, 81, V261-V270.
import numpy as np
import matplotlib.pyplot as plt
import pyseisdl as dl #dl: dictionary learning
import time
## generate the synthetic data


def gensyn(noise=False,seed=202122,var=0.2):
	'''
	gensyn: quickly generate the representative synthetic data used in the paper
	
	INPUT
	noise: if add noise
	seed: random number seed
	var: noise variance (actually the maximum amplitude of noise)
	
	OUTPUT
	case 1. data: clean synthetic data
	case 2. data,noisy: noisy case
		
	EXAMPLE 1
	from pyseistr import gensyn
	data=gensyn();
	import matplotlib.pyplot as plt;
	plt.imshow(data);plt.ylabel('Time sample');plt.xlabel('Trace');plt.show();

	EXAMPLE 2
	from pyseistr import gensyn
	data,noisy=gensyn(noise=True);
	import matplotlib.pyplot as plt;
	plt.subplot(1,2,1);plt.imshow(data,clim=[-0.2,0.2],aspect='auto');plt.xlabel('Trace');plt.ylabel('Time sample');
	plt.subplot(1,2,2);plt.imshow(noisy,clim=[-0.2,0.2],aspect='auto');plt.xlabel('Trace');
	plt.show();
	
	REFERENCE
	This data was originally from
	Wang, H., Chen, Y., Saad, O.M., Chen, W., Oboué, Y.A.S.I., Yang, L., Fomel, S. and Chen, Y., 2022. A MATLAB code package for 2D/3D local slope estimation and structural filtering. Geophysics, 87(3), pp.F1-F14.
	
	'''
	import numpy as np
	[w,tw]=ricker(30,0.001,0.1);
	t=np.zeros([300,1000]);sigma=300;A=100;B=200;
	data=np.zeros([400,1000]);

	[m,n]=t.shape;
	for i in range(1,n+1):
		k=np.floor(-A*np.exp(-np.power(i-n/2,2)/np.power(sigma,2))+B);k=int(k);
		if k>1 and k<=m:
			t[k-1,i-1]=1;

	for i in range(1,n+1):
		data[:,i-1]=np.convolve(t[:,i-1],w);
	
	if noise:
		data=data/np.max(np.max(data));
		np.random.seed(seed);
		noise=(np.random.rand(data.shape[0],data.shape[1])*2-1)*var;
		noisy=data+noise
		return data,noisy
	else:
		return data
	

def ricker(f,dt,tlength=None):
	'''
	ricker: Ricker wavelet of central frequency f.
	
	INPUT:
	f : central freq. in Hz (f <<1/(2dt) )
	dt: sampling interval in sec
	tlength : the duration of wavelet in sec
	
	OUTPUT: 
	w:  the Ricker wavelet
	tw: time axis
	
	EXAMPLE
	from pyseistr import ricker;
	wav,tw=ricker(20,0.004,2);
	import matplotlib.pyplot as plt;
	plt.plot(tw,wav);plt.xlabel('Time (s)');plt.ylabel('Amplitude');plt.show();
	
	'''
	import numpy as np
	
	if tlength!=None:
		nw=np.floor(tlength/dt)+1;
	else:
		nw=2.2/f/dt;
		nw=2*np.floor(nw/2)+1;
	nc=np.floor(nw/2);
	nw=int(nw)
	w =np.zeros(nw);
	
	k=np.arange(1,nw+1,1);
	alpha = (nc-k+1)*f*dt*np.pi;
	beta=np.power(alpha,2);
	w = (1.-beta*2)*np.exp(-beta);
	tw = -(nc+1-k)*dt;
	return w,tw
	

dc,dn=gensyn(noise=True);

## denoise using SGK %when K=3, SGK is even better than KSVD (13.23 dB)
# the computational difference is larger when T is larger
param={'T':2,'niter':10,'mode':1,'K':64};
mode=1;l1=4;l2=4;l3=1;s1=2;s2=2;s3=1;perc=1; 
t1 = time.time()
[d1,D,G,dct]=dl.sgk_denoise(dn,mode,[l1,l2,l3],[s1,s2,s3],perc,param);
t2 = time.time()
print('SGK takes %.2g seconds'%(t2-t1));

## benchmark with KSVD
t1 = time.time()
param={'T':2,'niter':10,'mode':1,'K':64};
[d2,D2,G2,dct2]=dl.ksvd_denoise(dn,mode,[l1,l2,l3],[s1,s2,s3],perc,param);
t2 = time.time()
print('KSVD takes %.2g seconds'%(t2-t1));

noi1=dn-d1;
noi2=dn-d2;

## compare with matlab
# import scipy
# from scipy import io
# datas = {"dc":dc, "dn":dn, "d1": d1, "d2": d2}
# scipy.io.savemat("sgk2d.mat", datas)

## compare SNR
print('SNR of Noisy is %g'%dl.snr(dc,dn,1));
print('SNR of SGK is %g'%dl.snr(dc,d1,1));
print('SNR of KSVD is %g'%dl.snr(dc,d2,1));

## plotting
fig = plt.figure(figsize=(8, 7))
ax=fig.add_subplot(3, 2, 1)
plt.imshow(dn,cmap='jet',clim=(-0.1, 0.1),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Noisy data');
ax=fig.add_subplot(3, 2, 3)
plt.imshow(d1,cmap='jet',clim=(-0.1, 0.1),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Denoised (SGK, SNR=%.4g dB)'%dl.snr(dc,d1,2));
ax=fig.add_subplot(3, 2, 4)
plt.imshow(noi1,cmap='jet',clim=(-0.1, 0.1),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Noise (SGK)');
ax=fig.add_subplot(3, 2, 5)
plt.imshow(d2,cmap='jet',clim=(-0.1, 0.1),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Denoised (KSVD, SNR=%.4g dB)'%dl.snr(dc,d2,2));
ax=fig.add_subplot(3, 2, 6)
plt.imshow(noi2,cmap='jet',clim=(-0.1, 0.1),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Noise (KSVD)');
plt.savefig('test_pyseisdl_sgk3d.png',format='png',dpi=300);
plt.show()




