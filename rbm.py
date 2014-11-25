import theano
import numpy
import helpers

class RBM:
    #in truth inputs are visible neurons and outputs hidden
    def __init__(self,  input, inputs_no, outputs_no):
        random_num_gen = numpy.random.RandomState()
        random_sym_gen = theano.tensor.shared_randomstreams()
        
        #for tanh
        self.W = theano.shared(value = helpers.initialized_weight_for_tan(
            random_num_gen, inputs_no, outputs_no),
            dtype=theano.config.floatX)


