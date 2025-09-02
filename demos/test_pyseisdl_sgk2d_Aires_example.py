#  DEMO script (python version) for pyseisdl
#  
#  KSVD will take a while (e.g., 400 s) and SGK takes 4 s
#  But SNR of KSVD is much higher than SGK in this case because the data is too noisy
#
#  Copyright (C) 2025 Lucas Aires
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

def dict_2(din, l1, l2, n_atoms=16, plot=False):
    """
    Constrói um dicionário de patches extraídos de uma imagem fonte.

    Os patches são selecionados aleatoriamente e adicionados ao dicionário se
    sua "energia" for maior ou igual a de um patch de referência.

    Parâmetros:
    ----------
    din : np.ndarray
        A imagem 2D da qual os patches serão extraídos.
    l1 : int
        A altura (l1) de cada patch.
    l2: int
        A largura (l2) de cada patch.
    n_atoms : int, opcional
        O número de patches a serem selecionados para o dicionário.
    plot : bool, opcional
        Se True, exibe uma visualização dos patches do dicionário.

    Retorno:
    -------
    D : np.ndarray
        O dicionário final como um array 3D no formato (n_patches, altura, largura).
    """
    import math
    from sklearn.feature_extraction import image
    
    np.random.seed(42) 
    patch_dims = (l1, l2)

    atom_ref = din[310:310+l1,110:110+l2]
    norm_ref = np.linalg.norm(atom_ref.T @ atom_ref, 'fro')

    selected_atoms_list = []
    
    max_attempts = n_atoms * 100 
    attempts = 0
    while len(selected_atoms_list) < n_atoms and attempts < max_attempts:
        atom = image.extract_patches_2d(din, patch_dims, max_patches=1).squeeze()
        
        if np.linalg.norm(atom.T @ atom, 'fro') >= norm_ref:
            selected_atoms_list.append(atom)
        attempts += 1
            
    if len(selected_atoms_list) < n_atoms:
        print(f"Aviso: Apenas {len(selected_atoms_list)} de {n_atoms} átomos foram encontrados após {max_attempts} tentativas.")

    D = np.array(selected_atoms_list)

    if plot and D.shape[0] > 0:
        n_plots = D.shape[0]
        n_cols = 8
        n_rows = math.ceil(n_plots / n_cols)
        
        plt.figure(figsize=(10, n_rows * 1.2))
        
        for i, patch in enumerate(D):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            ax.imshow(patch, cmap=plt.cm.gray, interpolation="nearest")
            ax.set_xticks(())
            ax.set_yticks(())

        plt.suptitle(
            f"Dicionário\n(Nº Átomos, Altura, Largura): {D.shape}",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return D

import numpy as np
import matplotlib.pyplot as plt
import pyseisdl as dl #dl: dictionary learning
import time

from sklearn.feature_extraction import image # add this library for getting the patches for the dictionary
# from dc_dn import dict_2 

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
# mode=1;l1=32;l2=16;l3=1;s1=1;s2=1;s3=1;perc=1; 
mode=1;l1=32;l2=16;l3=1;s1=16;s2=8;s3=1;perc=1;

#### Add this function to set the atoms (from the data) for the dictionary D 
n_atoms = 32
D = dict_2(dn, l1, l2, n_atoms, plot=True)
D = D.reshape((-1, D.shape[1]))


param={'T':6,'niter':10,'mode':1,'K':n_atoms, 'D': D};

t1 = time.time()
[d1,D,G,dct]=dl.sgk_denoise(dn,mode,[l1,l2,l3],[s1,s2,s3],perc,param);
t2 = time.time()
print('SGK takes %.2g seconds'%(t2-t1));

## benchmark with KSVD
t1 = time.time()
param={'T':6,'niter':10,'mode':1,'K':n_atoms, 'D': D};
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
