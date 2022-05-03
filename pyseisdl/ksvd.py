def ksvd(X,param):
	# ksvd: KSVD algorithm
	# BY Yangkang Chen
	# Jan, 2020
	# Ported to Python in May, 2022
	# 
	# INPUT
	# X:     input training samples
	# param: parameter struct
	#   param.mode=1;   	#1: sparsity; 0: error
	#   param.niter=10; 	#number of SGK iterations to perform; default: 10
	#   param.D=DCT;    	#initial D
	#   param.T=3;      	#sparsity level
	#
	# OUTPUT
	# D:    learned dictionary
	# G:    sparse coefficients
	#
	# for X=DG
	# size of X: MxN
	# size of D: MxK
	# size of G: KxN
	#
	# DEMO: demos/test_ksvd_denoise.py



	return X

