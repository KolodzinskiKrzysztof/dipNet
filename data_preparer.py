import cPickle
import gzip
from random import shuffle
import theano
import os
import numpy


class DataPreparer:
    def __init__(self, file_or_directory):
        self.data_source = file_or_directory
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), file_or_directory)
#        if not os.path.exists(self.data_source):
#            raise Exception("no file or directory named " + self.data_source)
        if os.path.isfile(path):
            pickle_file = None

            if self.data_source.endswith('.pkl.gz'):
                pickle_file = gzip.open(self.data_source)
            elif self.data_source.endswith('.pkl'):
                pickle_file = open(self.data_source)
            else:
                print('not supported data format')
                return
            pickleset = cPickle.load(pickle_file)
            self.dataset = [(theano.shared(numpy.asarray(data[0], 
                                                         dtype=theano.config.floatX), borrow=True), 
                             theano.tensor.cast(theano.shared(numpy.asarray(data[1], 
                                                         dtype=theano.config.floatX), borrow=True), 'int32'))
                            for data in pickleset]
            print(self.dataset)
            pickle_file.close()
        else:
            print("please be aware that if images are not serialized properly it is not going to work")
            filename = raw_input('provide path were pickle should be store')
            if not filename.endswith('.pkl'):
                filename += '.pkl'
#            self.dataset = __dir2pickle(self, filename)


#    def __dir2pickle(self, filename):
#        for filename in shuffle(os.listdir(self.data_source)):
#            numpy.ndarray(numpy.asarray(Image.open(filename)))

            
