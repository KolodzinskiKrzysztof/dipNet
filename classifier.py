import theano
import numpy

class Classifier:
    def __init__(self, classification_function, loss_function, input, inputs_no, outputs_no):
        self.W = theano.shared(value=numpy.zeros((inputs_no, outputs_no),
                dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((outputs_no),
                 dtype=theano.config.floatX), name='b', borrow=True)

        self.computed_labels = classification_function(theano.tensor.dot(input, self.W) + self.b)

        self.output_labels = theano.tensor.argmax(self.computed_labels, axis=1)

        self.theta = [self.W, self.b]
        

