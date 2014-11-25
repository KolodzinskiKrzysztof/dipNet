import theano
import numpy

def initialized_weigt_for_tan(random_gen, inputs_no, outputs_no):
    amplithude = 4 * numpy.sqrt(6. / (inputs_no + outputs_no))
    return numpy.asarray(random_num_gen.uniform(
        low= -amplithude, high=amplithude, size=(inputs_no, outputs_no)),
        dtype = theano.config.floatX)


             
