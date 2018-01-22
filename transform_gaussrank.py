#!/usr/bin/env python

"transform kin8nm train, validation and test sets with gaussrank"

import pandas as pd

from gaussrank import *

train_file = 'kin8nm/train.csv'
val_file = 'kin8nm/validation.csv'
test_file = 'kin8nm/test.csv'

train_output_file = 'kin8nm/train_gaussrank.csv'
val_output_file = 'kin8nm/validation_gaussrank.csv'
test_output_file = 'kin8nm/test_gaussrank.csv'

#

train = pd.read_csv( train_file, header = None )
val = pd.read_csv( val_file, header = None )
test = pd.read_csv( test_file, header = None )

d = pd.concat(( train, val, test ))

x_cols = d.columns[:-1]
x = d[x_cols]

s = GaussRankScaler()
x_ = s.fit_transform( x )
assert x_.shape == x.shape

d[x_cols] = x_

train_ = d[:len( train )]
assert len( train_ ) == len( train )

val_ = d[len( train ):len( train ) + len( val )]
assert len( val ) == len( val_ )

test_ = d[-len( test ):]
assert len( test ) == len( test_ )

train_.to_csv( train_output_file, index = None, header = None )
val_.to_csv( val_output_file, index = None, header = None )
test_.to_csv( test_output_file, index = None, header = None )
