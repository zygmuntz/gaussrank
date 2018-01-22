#!/usr/bin/env python

"[a beginning of] scikit-learn compatible implementation of GaussRank"

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfinv

class GaussRankScaler():

	def __init__( self ):
		self.epsilon = 0.001
		self.lower = -1 + self.epsilon
		self.upper =  1 - self.epsilon
		self.range = self.upper - self.lower

	def fit_transform( self, X ):
	
		i = np.argsort( X, axis = 0 )
		j = np.argsort( i, axis = 0 )

		assert ( j.min() == 0 ).all()
		assert ( j.max() == len( j ) - 1 ).all()
		
		j_range = len( j ) - 1
		self.divider = j_range / self.range
		
		transformed = j / self.divider
		transformed = transformed - self.upper
		transformed = erfinv( transformed )
		
		return transformed
		
# just an example of output		
if __name__ == '__main__':
	# simulating normalized ranks from -0.99 to 0.99
	x = np.arange( -0.99, 1, 0.01 )
	y = erfinv( x )
	
	plt.hist( y )
		