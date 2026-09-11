from copy import deepcopy

import numpy as np
from metrics import multiclass_accuracy
from tqdm import tqdm


class Dataset:
    """
    Utility class to hold training and validation data
    """

    def __init__(self, train_X, train_y, val_X, val_y):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y


class Trainer:
    """
    Trainer of the neural network models
    Perform mini-batch SGD with the specified data, model,
    training parameters and optimization rule
    """

    def __init__(self, model, dataset, optim,
                 num_epochs=20,
                 batch_size=20,
                 learning_rate=1e-2,
                 learning_rate_decay=1.0):
        """
        Initializes the trainer

        Arguments:
        model - neural network model
        dataset, instance of Dataset class - data to train on
        optim - optimization method (see optim.py)
        num_epochs, int - number of epochs to train
        batch_size, int - batch size
        learning_rate, float - initial learning rate
        learning_rate_decal, float - ratio for decaying learning rate
           every epoch
        """
        self.dataset = dataset
        self.model = model
        self.optim = optim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.learning_rate_decay = learning_rate_decay

        self.optimizers = None

    def setup_optimizers(self):
        params = self.model.params() # get params dict
        self.optimizers = {} # new dict params for optim
        for param_name, param in params.items(): # gogo through params model dict items
            self.optimizers[param_name] = deepcopy(self.optim) # for every param creates new SGD item?? idk

    def compute_accuracy(self, X, y):
        """
        Computes accuracy on provided data using mini-batches
        """
        indices = np.arange(X.shape[0]) # arrange len featers
        sections = np.arange(self.batch_size, X.shape[0], self.batch_size) # arrange sections batch
        batches_indices = np.array_split(indices, sections) # split on sections

        pred_batch = np.zeros((self.batch_size, 10)) # create zeros arrya for pred
        y_batch = np.zeros((self.batch_size, 10))

        acc_list = []
        for batch_indices in batches_indices:
            batch_X = X[batch_indices]
            pred_batch = self.model.predict(batch_X)
            y_batch = y[batch_indices]

            acc_list.append(multiclass_accuracy(pred_batch, y_batch))

        return np.mean(acc_list)
    
    def fit(self):
        """
        Trains a model
        """
        if self.optimizers is None:
            self.setup_optimizers()

        num_train = self.dataset.train_X.shape[0]

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        for epoch in range(self.num_epochs):
            shuffled_indices = np.arange(num_train) # just generate arrange len dataset
            np.random.shuffle(shuffled_indices) # shuffle
            sections = np.arange(self.batch_size, num_train, self.batch_size) # generate arrange len dataset with step batch_szie
            batches_indices = np.array_split(shuffled_indices, sections) # split by indices from sections

            batch_losses = []
            batch_loop = tqdm(batches_indices, leave=False)
            for batch_indices in batch_loop:
                # TODO Generate batches based on batch_indices and
                # use model to generate loss and gradients for all
                # the params
                train_X_batch = self.dataset.train_X[batch_indices]
                train_y_batch = self.dataset.train_y[batch_indices]
                loss = self.model.compute_loss_and_gradients(train_X_batch, train_y_batch)


                for param_name, param in self.model.params().items():
                    optimizer = self.optimizers[param_name]
                    param.value = optimizer.update(param.value, param.grad, self.learning_rate)

                batch_losses.append(loss)

            if np.not_equal(self.learning_rate_decay, 1.0): # self.learning_rate_decay != 1.0
                self.learning_rate *= self.learning_rate_decay
                

            ave_loss = np.mean(batch_losses)

            train_accuracy = self.compute_accuracy(self.dataset.train_X,
                                                   self.dataset.train_y)

            val_accuracy = self.compute_accuracy(self.dataset.val_X,
                                                 self.dataset.val_y)

            # print("Loss: %f, Train accuracy: %f, val accuracy: %f" %
                #   (batch_losses[-1], train_accuracy, val_accuracy))
            batch_loop.set_description("Loss: %f, Train accuracy: %f, val accuracy: %f" %
                                        (batch_losses[-1], train_accuracy, val_accuracy))
            loss_history.append(ave_loss)
            train_acc_history.append(train_accuracy)
            val_acc_history.append(val_accuracy)

        return loss_history, train_acc_history, val_acc_history
