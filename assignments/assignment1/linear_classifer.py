import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    assert len(predictions.shape) in {1, 2}, 'Bad dimension in predictions!'
    if len(predictions.shape) == 2:
        shifted_predictions = predictions - np.max(predictions, axis=1, keepdims=True)
        exp_predictions = np.exp(shifted_predictions)
        probs = exp_predictions / np.sum(exp_predictions, axis=1, keepdims=True)
        return probs
    
    shifted_predictions = predictions - np.max(predictions)
    exp_predictions = np.exp(shifted_predictions)
    probs = exp_predictions / np.sum(exp_predictions)
    return probs

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    assert len(probs.shape) in {1, 2}, 'Bad dimension in probs!'
    if len(probs.shape) == 2:
        batch_size = probs.shape[0]
        row_indices = np.arange(batch_size)
        class_indices = target_index.reshape(-1)
        correct_probs = probs[row_indices, class_indices]
        return -np.mean(np.log(correct_probs))
        
    return -np.log(probs[target_index])

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()

    if len(dprediction.shape) == 2:
      batch_size = probs.shape[0]
      row_indices = np.arange(batch_size)
      class_indices = target_index.reshape(-1)
      dprediction[row_indices, class_indices] -= 1 
      dprediction /= batch_size
    else:
        dprediction[target_index] -= 1

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.linalg.norm(W[:-1,], ord='fro') ** 2 
    grad = 2 * reg_strength * W
    grad[-1,] = 0
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = X @ W # (num_batch, classes)
    loss, dpredictinos = softmax_with_cross_entropy(predictions, target_index)
    dW = X.T @ dpredictinos
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(num_train)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            loss = 0
            for i_batch in batches_indices:
                X_batch = X[i_batch]
                y_batch = y[i_batch]
                loss_sm, dW_sm = linear_softmax(X_batch, self.W, y_batch)
                loss_l2, dl2 = l2_regularization(self.W, reg_strength=reg)
                dW = dW_sm + dl2
                self.W -= learning_rate * dW
                loss += (loss_sm + loss_l2)

            loss /= len(batches_indices)
            loss_history.append(loss)
            # end
            # print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        pred = X @ self.W
        y_pred = np.argmax(pred, axis=1)

        return y_pred



                
                                                          

            

                
