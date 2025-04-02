import numpy as np

class Optimizer:
    def __init__(self):
        raise NotImplementedError(f'{self.__class__.__name__} is not implemented.')
    
    def prev_update(self):
        raise NotImplementedError('prev_update() is not implemented.')
    
    def update_params(self, grads, i):
        raise NotImplementedError('update_params() is not implemented.')
    
    def step(self):
        raise NotImplementedError('step() is not implemented.')
    
class Adam(Optimizer):
    def __init__(self, lr=0.001, decay=0, betas=(0.9, 0.999), epsilon=1e-8):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.epsilon = epsilon
        self.t = 1
        self.m = []
        self.v = []

    def init_cache(self, len_params):
        self.m = [0] * len_params
        self.v = [0] * len_params
    
    def prev_update(self):
        if self.decay:
            self.current_lr = self.lr * (1 / (1 + self.decay * self.t))

    def update_params(self, grads, i):
        self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grads
        self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * grads**2
        m_h = self.m[i] / (1 - self.beta_1**self.t)
        v_h = self.v[i] / (1 - self.beta_2**self.t)
        return self.current_lr * (m_h / (np.sqrt(v_h) + self.epsilon))
    
    def step(self):
        self.t += 1







class Loss:
    def __init__(self, mode='sum'):
        modes = 'sum', 'mean'
        mode = mode.lower()
        if mode in modes:
            self.mode = mode
        else:
            raise ValueError(f'{mode} doesn\'t exists in {modes}')
        
    def forward(self, y_true, y_pred):
        raise NotImplementedError('forward() is not implemented.')
    
    def __call__(self, y_true, y_pred, derv=False):
        func = np.sum if self.mode == 'sum' else np.mean
        if derv: return self.backward(y_true, y_pred)
        return func(self.forward(y_true, y_pred))

    def backward(self, y_true, y_pred):
        raise NotImplementedError('backward() is not implemented.')

class CategoricalCrossEntropy(Loss):
    def __init__(self, mode='sum', softmax=True, epsilon=1e-8):
        super().__init__(mode)
        self.softmax = softmax
        self.epsilon = epsilon

    def forward(self, y_true, y_pred):
        return -y_true*np.log(y_pred+self.epsilon)
    
    def backward(self, y_true, y_pred):
        return y_pred-y_true if self.softmax else -y_true/(y_pred+self.epsilon)

class BinaryCrossEntropy(Loss):
    def __init__(self, mode='sum', epsilon=1e-8):
        super().__init__(mode)
        self.epsilon = epsilon

    def forward(self, y_true, y_pred):
        return -y_true * np.log(y_pred + self.epsilon) - (1 - y_true) * np.log(1 - y_pred + self.epsilon)
    
    def backward(self, y_true, y_pred):
        return -y_true/(y_pred + self.epsilon) + (1 - y_true)/(1 - y_pred + self.epsilon)






class Accuracy:
    def __init__(self, mode='sum'):
        modes = 'sum', 'mean'
        mode = mode.lower()
        if mode in modes:
            self.mode = mode
        else:
            raise ValueError(f'{mode} doesn\'t exists in {modes}')

    def call(self, y_true, y_pred):
        raise NotImplementedError('call() is not implemented.')
    
    def __call__(self, y_true, y_pred):
        func = np.sum if self.mode == 'sum' else np.mean
        return func(self.call(y_true, y_pred))

class CategoricalAccuracy(Accuracy):
    def call(self, y_true, y_pred):
        return np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)
    
class BinaryAccuracy(Accuracy):
    def call(self, y_true, y_pred):
        return y_true == y_pred
    
class FactorBinaryAccuracy(Accuracy):
    def __init__(self, mode='sum', factor=0.5):
        super().__init__(mode)
        self.factor = factor
    
    def call(self, y_true, y_pred):
        return (y_pred > self.factor) == y_true
    

class FactorCategoricalAccuracy(Accuracy):
    def __init__(self, mode='sum', factor=0.5):
        super().__init__(mode)
        self.factor = factor

    def call(self, y_true, y_pred):
        return (np.argmax(y_pred, axis=1) > self.factor) == np.argmax(y_pred, axis=1)






class Activation:
    def forward(self, x):
        raise NotImplementedError('forward() is not implemented.')
    
    def backward(self, x):
        raise NotImplementedError('backward() is not implemented.')
    
    def __call__(self, x, derv=False):
        if derv: return self.backward(x)
        return self.forward(x)
    
class ReLU(Activation):
    def forward(self, x):
        return np.maximum(x, 0)
    
    def backward(self, x):
        return np.where(x>0, 1, 0)
    
class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)
    
class Softmax(Activation):
    def __init__(self, last_layer=True):
        self.last_layer = last_layer

    def forward(self, x):
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        s = x_exp / np.sum(x_exp, axis=1, keepdims=True)
        return s
    
    def backward(self, x):
        if self.last_layer: return 1
        s = self.forward(x)
        return s * (1 - s)

class DNN:
    count = 0
    def __init__(self, neurons: tuple, activations: tuple, name=None):
        if name:
            self.name = name
        else:
            self.name = f'{self.__class__.__name__}_{__class__.count}'
        self.activations = activations
        self.n_layers = len(neurons)-1
        self.W = [np.random.randn(neurons[i], neurons[i+1]) * np.sqrt(2/neurons[i]) for i in range(self.n_layers)]
        self.b = [np.zeros((1, neurons[i])) for i in range(1, self.n_layers+1)]
        self.z = [0] * self.n_layers
        self.a = [0] * (self.n_layers+1)
        self.dW = [0] * self.n_layers
        self.db = [0] * self.n_layers
        self.compiled = False
        __class__.count += 1

    def compile(self, loss: Loss, optimizer: Optimizer, accuracy: Accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer.init_cache(self.n_layers*2)
        self.accuracy = accuracy
        self.compiled = True
    
    def forward(self, x):
        self.a[0] = x

        for l in range(self.n_layers):
            self.z[l] = self.a[l] @ self.W[l] + self.b[l]
            self.a[l+1] = self.activations[l](self.z[l])

        return self.a[-1]

    def update_params(self):
        self.optimizer.prev_update()
        for i in range(self.n_layers):
            self.W[i] -= self.optimizer.update_params(self.dW[i], i*2)
            self.b[i] -= self.optimizer.update_params(self.db[i], i*2+1)
        self.optimizer.step()

    def backward(self, y=None, out=None, dout=None, learn=True):
        if y is not None:
            dL = self.loss(y, out, derv=True)
        elif dout is not None:
            dL = dout
        else: raise ValueError('y and dout are None')
        dL *= self.activations[-1](self.z[-1], derv=True)
        m = dL.shape[0]

        for l in range(self.n_layers-1, -1, -1):
            self.dW[l] = self.a[l].T @ dL / m
            self.db[l] = np.sum(dL, axis=0, keepdims=True) / m
            if l > 0: dL = dL @ self.W[l].T * self.activations[l-1](self.z[l-1], derv=True)

        dout = dL @ self.W[0].T

        if learn:
            self.update_params()

        return dout
    
    def train(self, x, y, epochs=10, batch_size=32, verbose=True, print_every=2, x_test=None, y_test=None):
        if not self.compiled: raise ValueError(f'Model {self.name} is not compiled.')
        num_batches = int(np.ceil(x.shape[0] / batch_size))
        for ep in range(1, epochs+1):

            total_loss = 0.0
            total_acc = 0.0

            for idx, b in enumerate(range(0, x.shape[0], batch_size), start=1):
                x_batch = x[b:b+batch_size]
                y_batch = y[b:b+batch_size]

                predictions = self.forward(x_batch)
                total_loss += self.loss(y_batch, predictions)
                total_acc += self.accuracy(y_batch, predictions)

                self.backward(y_batch, predictions, learn=True)
                print(f'Epoch: {ep} | Batch: {idx}/{num_batches}', end='\r')
            avg_loss = total_loss / x.shape[0]
            avg_acc = total_acc / x.shape[0]

            if verbose and ep % print_every == 0:
                message = f'Epoch: {ep}, Loss: {avg_loss}, Acc: {avg_acc}'
                if x_test is not None and y_test is not None:
                    
                    total_loss = 0.0
                    total_acc = 0.0

                    for b in range(0, x_test.shape[0], batch_size):
                        x_batch = x_test[b:b+batch_size]
                        y_batch = y_test[b:b+batch_size]

                        predictions = self.forward(x_batch)
                        total_loss += self.loss(y_batch, predictions)
                        total_acc += self.accuracy(y_batch, predictions)

                    message += f', Test Loss: {total_loss/x_test.shape[0]}, Test Acc: {total_acc/x_test.shape[0]}'
                print(message)