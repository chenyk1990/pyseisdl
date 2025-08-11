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


### Load the data from Lucas Aires (https://github.com/aaspip/data/blob/main/diff2_cutTmax2s_SR0p5.su)
import obspy
dc = np.array(obspy.read("diff2_cutTmax2s_SR0p5.su")).T
[n1,n2]=dc.shape
np.random.seed(201415)
noise=0.2*np.random.randn(n1,n2)  ## multiplicador aumenta o ruído
dn = dc + noise	
### Load the data from Lucas Aires

## denoise using SGK %when K=3, SGK is even better than KSVD (13.23 dB)
# the computational difference is larger when T is larger
param={'T':6,'niter':10,'mode':1,'K':64};
mode=1;l1=4;l2=4;l3=1;s1=2;s2=2;s3=1;perc=1; 
t1 = time.time()
[d1,D,G,dct]=dl.sgk_denoise(dn,mode,[l1,l2,l3],[s1,s2,s3],perc,param);
t2 = time.time()
print('SGK takes %.2g seconds'%(t2-t1));

## benchmark with KSVD
t1 = time.time()
param={'T':6,'niter':10,'mode':1,'K':64};
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




