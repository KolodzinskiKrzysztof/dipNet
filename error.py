import theano

def binary(computed_labels, correct_labels):
    return theano.tensor.mean(theano.tensor.neq(computed_labels, correct_labels))
