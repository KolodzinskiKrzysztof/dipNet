import argparse
import data_preparer
import theano
import classifier
import os
import loss_function
import training_function
import numpy

#parser = argparse.ArgumentParser(description = 'give me dataset')
#parser.add_argument('dataset', metavar='D', nargs=1, help='file with training examples')

#args = parser.parse_args()

training_set, validation_set, test_set = data_preparer.DataPreparer('./mnist.pkl.gz').dataset
#dataset - training, valid, test

#computed_classification - 2d matrix - row per example, column pre class

def binary_errors(computed_labels, correct_labels):
    return theano.tensor.mean(theano.tensor.neq(computed_labels, correct_labels))

x = theano.tensor.matrix('x')
y = theano.tensor.ivector('y')

classifier = classifier.Classifier(theano.tensor.nnet.softmax,  x, 28*28, 10)

cost = loss_function.negative_log_likehood(classifier.computed_labels, y)


W_gradient = theano.tensor.grad(cost = cost, wrt=classifier.W)
b_gradient = theano.tensor.grad(cost = cost, wrt=classifier.b)

learning_rate = 0.13

updates = [(classifier.W, classifier.W - learning_rate * W_gradient),
           (classifier.b, classifier.b - learning_rate * b_gradient)]

index = theano.tensor.iscalar()
batch_size=600
n_train_batches = training_set[0].get_value(borrow=True).shape[0]/batch_size
n_test_batches = test_set[0].get_value(borrow=True).shape[0]/batch_size
n_validation_batches = validation_set[0].get_value(borrow=True).shape[0]/batch_size

error = binary_errors(classifier.output_labels, y)

train_model = training_function.training(index, cost, updates, training_set, batch_size,x,y, updates)
validate = training_function.test(index, error, validation_set, batch_size,x,y)
test = training_function.test(index, error, test_set, batch_size,x,y)

n_epochs = 1000
validation_freq = 2000
max_not_learning_trainings = 5

best_validation_error = numpy.inf
break_loop = False

for epoch in xrange(n_epochs):
    for minibatch_index in xrange(n_train_batches):
        iteration = epoch*n_train_batches + minibatch_index
        print "iteration " + str(iteration)
        minibatch_cost = train_model(minibatch_index)
        if (iteration % validation_freq ==0):
            validation_losses  = [validate(i) for i in xrange(n_validation_batches)]
            current_validation_error = numpy.mean(validation_losses)
            print "current validation error" + str (current_validation_error)
            if (current_validation_error < best_validation_error):
                best_validation_error = current_validation_error
                not_learning_trainings = 0
                test_losses = [test(i) for i in xrange(n_test_batches)]
                print "test loses" + str(numpy.mean(test_losses))
            else:
                not_learning_trainings += 1
        if (not_learning_trainings >= max_not_learning_trainings):
            break_loop = True
            break
    if (break_loop):
        break



