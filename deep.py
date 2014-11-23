import argparse
import data_preparer
import theano
import classifier
import os


#parser = argparse.ArgumentParser(description = 'give me dataset')
#parser.add_argument('dataset', metavar='D', nargs=1, help='file with training examples')

#args = parser.parse_args()

training_set, validation_set, test_set = data_preparer.DataPreparer('./mnist.pkl.gz').dataset
#dataset - training, valid, test

#computed_classification - 2d matrix - row per example, column pre class
def negative_log_likehood(computed_classifications, correct_labels):
    theano.tensor.log(computed_classifications)
    computed_classifications[theano.tensor.arange(correct_labels.shape[0]), correct_labels]
    theano.tensor.log(computed_classifications)[theano.tensor.arange(correct_labels.shape[0]), correct_labels]
    return -theano.tensor.mean(theano.tensor.log(computed_classifications)[theano.tensor.arange(correct_labels.shape[0]), correct_labels])

def binary_errors(computed_labels, correct_labels):
    theano.tensor.mean(theano.tensor.neq(computed_labels, correct_labels))

x = theano.tensor.matrix('x')
y = theano.tensor.ivector('y')

classifier = classifier.Classifier(theano.tensor.nnet.softmax, negative_log_likehood, x, 28*28, 10)

cost = negative_log_likehood(classifier.computed_labels, y)


W_gradient = theano.tensor.grad(cost = cost, wrt=classifier.W)
b_gradient = theano.tensor.grad(cost = cost, wrt=classifier.b)

learning_rate = 0.13

updates = [(classifier.W, classifier.W - learning_rate * W_gradient),
           (classifier.b, classifier.b - learning_rate * b_gradient)]

index = theano.tensor.iscalar()
batch_size=600
n_train_batches = training_set[0].get_value(borrow=True).shape[0]/batch_size

train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: training_set[0][index * batch_size: (index + 1) * batch_size],
            y: training_set[1][index * batch_size: (index + 1) * batch_size]
        }
    )

for minibatch_index in xrange(n_train_batches):
     avgcost = train_model(minibatch_index)




