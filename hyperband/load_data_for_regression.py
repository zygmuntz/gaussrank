import numpy as np

# comment out as needed

# original
train_file = 'data/kin8nm/train.csv'
valid_file = 'data/kin8nm/validation.csv'
test_file = 'data/kin8nm/test.csv'

# gaussrank version
train_file = 'data/kin8nm/train_gaussrank.csv'
valid_file = 'data/kin8nm/validation_gaussrank.csv'
test_file = 'data/kin8nm/test_gaussrank.csv'

# random projections version
train_file = 'data/kin8nm/train_random_projections.csv'
valid_file = 'data/kin8nm/validation_random_projections.csv'
test_file = 'data/kin8nm/test_random_projections.csv'

print "loading {} and {}...".format( train_file, valid_file )

train = np.loadtxt( open( train_file ), delimiter = "," )
valid = np.loadtxt( open( valid_file ), delimiter = "," )
#test = np.loadtxt( open( test_file ), delimiter = "," )

y_train = train[:,-1]
y_test = valid[:,-1]
#y_test = test[:,-1]

x_train = train[:,0:-1]
x_test = valid[:,0:-1]
#x_test = test[:,0:-1]

data = { 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test }
