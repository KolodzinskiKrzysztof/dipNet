import theano

def negative_log_likehood(computed_classifications, correct_labels):
    return -theano.tensor.mean(theano.tensor.log(computed_classifications)[theano.tensor.arange(correct_labels.shape[0]), correct_labels])

