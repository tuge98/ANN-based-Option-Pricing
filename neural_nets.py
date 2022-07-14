from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

class MLPnetwork:

    def __init__(self, num_epochs, batch,nodes,X_train, y_train,X_test, y_test):
        self.num_epochs = num_epochs
        self.batch = batch
        self.nodes = nodes
        self.model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    

    def initNetwork(self):
        sess = tf.compat.v1.InteractiveSession()
        sess.run(tf.compat.v1.global_variables_initializer())
        self.model = Sequential()
        self.model.add(Dense(self.nodes, input_dim=self.X_train.shape[1]))
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.25))

        self.model.add(Dense(self.nodes, activation='elu'))
        self.model.add(Dropout(0.25))

        
        self.model.add(Dense(self.nodes, activation='relu'))
        self.model.add(Dropout(0.25))


        self.model.add(Dense(1))
        self.model.add(Activation(self.custom_activation))
        self.model.compile(loss='mse', optimizer='rmsprop')

        # fitting neural network


        return self.model
    
    def outSample(self):
        self.model.fit(self.X_train, self.y_train, batch_size=self.batch,

        epochs=self.num_epochs, validation_split=0.1, verbose=2)
        
        y_test_hat = self.model.predict(self.X_test)
        #y_test_hat = np.squeeze(y_test_hat)
        return y_test_hat

    def obtainGradients(self):
        sess = tf.compat.v1.InteractiveSession()
        sess.run(tf.compat.v1.global_variables_initializer())
        gradients = tf.gradients(self.model.output[:, 0], self.model.input)
        evaluated_gradients_1 = sess.run(gradients[0], feed_dict={self.model.input: self.X_test})
        print(evaluated_gradients_1)
        


        def Extract(lst):
            return [item[0] for item in lst]


        deltaANN = Extract(evaluated_gradients_1)

        return deltaANN
    

    def custom_activation(self,x):
        return backend.exp(x)