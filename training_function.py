import theano

def training(index, cost_fn, updates_fn, dataset, batch_size, x, y, updates):
    return theano.function(
        inputs=[index],
        outputs=cost_fn,
        updates=updates_fn,
        givens={
            x: dataset[0][index * batch_size: (index + 1) * batch_size],
            y: dataset[1][index * batch_size: (index + 1) * batch_size]
        }
    )

def test(index, error_fn, dataset, batch_size, x, y):
    return theano.function(
        inputs=[index],
        outputs=error_fn,
        givens={
            x: dataset[0][index * batch_size: (index + 1) * batch_size],
            y: dataset[1][index * batch_size: (index + 1) * batch_size]
        }
    )

