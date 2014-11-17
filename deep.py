import argparse
import data_preparer
import theano


#parser = argparse.ArgumentParser(description = 'give me dataset')
#parser.add_argument('dataset', metavar='D', nargs=1, help='file with training examples')

#args = parser.parse_args()

dataset = DataPreparer('minst.pkl.gz').dataset
#dataset - training, valid, test

