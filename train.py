import tensorflow as tf


# the updated TrainModel class, with l2 regularization and gradient clipping added:
class TrainModelUpdated:

  def __init__(self, model, batch_size = 8, lr = 0.001, loss = tf.keras.losses.MeanSquaredError, opt=tf.keras.optimizers.Adam, 
               use_l2_reg=False, use_clipnorm=False, reg_weig_lambda=0.001):

    self.model      = model
    self.loss       = loss()
    self.use_l2_reg = use_l2_reg
    self.reg_weig_lambda = reg_weig_lambda
  
    
    #### Gradient clipping helps to avoid exploding gradients:    
    # Setting the clipnorm parameter to 0.2, which means that in case the 
    # gradient vector norm will exceed 0.2, the values in the vector will 
    #be rescaled such that the norm will be 0.2
    if use_clipnorm:
      self.optimizer  = opt(learning_rate = lr, clipnorm=0.2)
    ##########
    else:
      self.optimizer  = opt(learning_rate = lr)


    self.batch_size = batch_size

    self.train_loss     = tf.keras.metrics.Mean(name='train_loss')
    self.test_loss     = tf.keras.metrics.Mean(name='test_loss')
  
  def apply_loss_with_l2(self, targets, predictions, llambda=0.002):
    """ computes the loss with the l2 regularization """
    weights   = self.model.trainable_variables
    # add_n will add all the input tensors element-wise.
    # l2_loss will compute the l2 norm of a vector: https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    l2_regularization = tf.add_n([ tf.nn.l2_loss(var) for var in weights ]) * llambda
    initial_loss = self.loss(targets, predictions)
    return initial_loss + llambda * l2_regularization

  
  @tf.function
  def train_step(self, x , y):
    with tf.GradientTape() as tape:
      predictions = self.model(x)
      if self.use_l2_reg:
        loss = self.apply_loss_with_l2(y, predictions, self.reg_weig_lambda)
      else:
        loss = self.loss(y, predictions)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    self.train_loss.update_state(loss)
    return loss

  @tf.function
  def test_step(self, x , y):
    predictions = self.model(x)
    if self.use_l2_reg:
        loss = self.apply_loss_with_l2(y, predictions, self.reg_weig_lambda)
    else:
        loss = self.loss(y, predictions)

    self.test_loss.update_state(loss)
    return loss

  def train(self):
    loss = []
    for bX, bY in self.train_ds:
      loss.append(self.train_step(bX, bY))
    return loss
  
  def test(self):
    loss = []
    for bX, bY in self.test_ds:
      loss.append(self.test_step(bX, bY))  
    return loss 
  
  def run(self, dataX, dataY, testX, testY, epochs, verbose=2):
    history = []
    
    self.train_ds = tf.data.Dataset.from_tensor_slices((dataX, dataY)).shuffle(16000).batch(self.batch_size)
    self.test_ds  = tf.data.Dataset.from_tensor_slices((testX,testY)).batch(self.batch_size)
    
    for i in range(epochs):
      
      train_loss = self.train()
      test_loss  = self.test()

      history.append([train_loss,test_loss])

      if verbose > 0 and (i==0 or (i+1)%5==0):
        print(f"epoch: {i+1}, TRAIN LOSS: {self.train_loss.result()},TEST LOSS: {self.test_loss.result()}")
        
        self.train_loss.reset_states()
        self.test_loss.reset_states()

    return history