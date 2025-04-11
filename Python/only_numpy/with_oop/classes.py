import numpy as np
from IPython.display import clear_output
import os
import matplotlib.pyplot as plt
import time
import glob
import imageio
from functions import *

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

    def copy(self):
        new_optim = Adam(lr=self.lr, decay=self.decay, betas=(self.beta_1, self.beta_2), epsilon=self.epsilon)
        return new_optim

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
    def __init__(self, mode='mean'):
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
    
class MeanSquaredError(Loss):
    def forward(self, y_true, y_pred):
        return (y_pred-y_true)**2
    
    def backward(self, y_true, y_pred):
        return 2*(y_pred-y_true)

class CategoricalCrossEntropy(Loss):
    def __init__(self, mode='mean', softmax=True, epsilon=1e-8):
        super().__init__(mode)
        self.softmax = softmax
        self.epsilon = epsilon

    def forward(self, y_true, y_pred):
        return -y_true*np.log(y_pred+self.epsilon)
    
    def backward(self, y_true, y_pred):
        return y_pred-y_true if self.softmax else -y_true/(y_pred+self.epsilon)

class BinaryCrossEntropy(Loss):
    def __init__(self, mode='mean', epsilon=1e-8):
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
    def __init__(self, mode='mean', factor=0.5):
        super().__init__(mode)
        self.factor = factor
    
    def call(self, y_true, y_pred):
        return (y_pred > self.factor) == y_true

class FactorCategoricalAccuracy(Accuracy):
    def __init__(self, mode='mean', factor=0.5):
        super().__init__(mode)
        self.factor = factor

    def call(self, y_true, y_pred):
        return (np.argmax(y_pred, axis=1) > self.factor) == np.argmax(y_true, axis=1)






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
    
class LeakyReLU(Activation):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x>0, x, x*self.alpha)
    
    def backward(self, x):
        return np.where(x>0, 1, self.alpha)
    
class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)
    
class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - np.tanh(x)**2
    
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

class Model:
    pass


class DNN(Model):
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

            avg_loss = total_loss / (x.shape[0] if self.loss.mode == 'sum' else num_batches)
            avg_acc = total_acc / (x.shape[0] if self.accuracy.mode == 'sum' else num_batches)

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

class RNN(Model):
    def __init__(self, input_size, hidden_size, output_size, last_activation=identity):
        self.Wx = np.random.randn(input_size, hidden_size) * np.sqrt(2/(input_size+hidden_size))
        self.Wh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1/hidden_size)
        self.Wy = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)

        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        self.last_activation = last_activation

        self.params = [self.Wy, self.by, self.Wh, self.bh, self.Wx]
        self.n_params = len(self.params)
    
    def compile(self, loss: Loss, optimizer: Optimizer, accuracy: Accuracy=None):
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer.init_cache(len(self.params))
        self.accuracy = accuracy
        self.compiled = True

    def update_params(self, grads):
        self.optimizer.prev_update()
        for i in range(self.n_params):
            self.params[i] -= self.optimizer.update_params(grads[i], i)
        self.optimizer.step()

    def forward_cell(self, xt, h_prev):
        return tanh(xt @ self.Wx + h_prev @ self.Wh + self.bh)
    
    def forward(self, x):
        T = x.shape[0]

        self.H = np.zeros((T+1, 1, self.Wh.shape[1]))
        Y = np.zeros((T, self.Wy.shape[1]))

        for t in range(T):
            self.H[t+1] = self.forward_cell(x[t:t+1], self.H[t])

            Y[t] = self.last_activation(self.H[t+1] @ self.Wy + self.by)

        return Y
    
    def backward_cell(self, dL, xt, ht, h_prev):
        dL *= derv_tanh(ht)

        dbh = np.sum(dL, axis=0, keepdims=True)

        dWx = xt.T @ dL
        dWh = h_prev.T @ dL

        dh_t = dL @ self.Wh.T
        dxt = dL @ self.Wx.T

        return dWx, dWh, dbh, dh_t, dxt
    
    def backward(self, x, y_true, y_pred, learn=True):
        T = x.shape[0]

        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dWy = np.zeros_like(self.Wy)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros((1, self.Wh.shape[1]))
        dx = np.zeros_like(x)

        for t in reversed(range(T)):
            xt = x[t:t+1]
            yt = y_true[t:t+1]
            outp = y_pred[t:t+1]
            h_prev = self.H[t]
            ht = self.H[t+1]

            dL = self.loss(yt, outp, derv=True)

            dWy += ht.T @ dL
            dby += np.sum(dL, axis=0, keepdims=True)

            dL = dL @ self.Wy.T * self.last_activation(ht, derv=True) + dh_next

            dWxt, dWht, dbht, dh_next, dx[t] = self.backward_cell(dL, xt, ht, h_prev)

            dWx += dWxt; dWh += dWht; dbh += dbht

        dWx /= T; dWh /= T; dWy /= T
        dbh /= T; dby /= T

        grads = dWy, dby, dWh, dbh, dWx

        if learn:
            self.update_params(grads)

        return dx
    
    def train(self, x, y, epochs=100, batch_size=8, print_every=0.1):
        losses = []
        for epoch in range(1, epochs+1):
            for batch in range(0, x.shape[0], batch_size):
                x_batch = x[batch:batch+batch_size]
                y_batch = y[batch:batch+batch_size]
                y_pred = self.forward(x_batch)

                self.backward(x_batch, y_batch, y_pred, learn=True)
            
            loss = self.loss(y, self.forward(x))
            losses.append(loss)
            if epoch % max(1, int(epochs*print_every)) == 0:
                print(f'Epoch: [{epoch}/{epochs}]> Loss: {loss}')
        return losses

class LSTM(Model):
    def __init__(self, input_size, hidden_size, output_size, last_activation=identity):
        self.Uf, self.Wf, self.bf = self._init_weights(input_size, hidden_size)
        self.Ui, self.Wi, self.bi = self._init_weights(input_size, hidden_size)
        self.Uo, self.Wo, self.bo = self._init_weights(input_size, hidden_size)
        self.Ug, self.Wg, self.bg = self._init_weights(input_size, hidden_size)

        self.Wy = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.by = np.zeros((1, output_size))

        self.last_activation = last_activation

        self.params = [self.Uf, self.Wf, self.bf,
                       self.Ui, self.Wi, self.bi,
                       self.Uo, self.Wo, self.bo,
                       self.Ug, self.Wg, self.bg,
                       self.Wy, self.by]
        
        self.n_params = len(self.params)

    def _init_weights(self, input_size, hidden_size):
        U = np.random.randn(input_size, hidden_size) * np.sqrt(2/(input_size+hidden_size))
        W = np.random.randn(hidden_size, hidden_size) * np.sqrt(1/hidden_size)
        b = np.zeros((1, hidden_size))
        return U, W, b
    
    def compile(self, loss: Loss, optimizer: Optimizer, accuracy: Accuracy=None):
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer.init_cache(len(self.params))
        self.accuracy = accuracy
        self.compiled = True

    def update_params(self, grads):
        self.optimizer.prev_update()
        for i in range(self.n_params):
            self.params[i] -= self.optimizer.update_params(grads[i], i)
        self.optimizer.step()

    def forward_cell(self, xt, h_prev, c_prev):
        ft = sigmoid(xt @ self.Uf + h_prev @ self.Wf + self.bf)
        it = sigmoid(xt @ self.Ui + h_prev @ self.Wi + self.bi)
        ot = sigmoid(xt @ self.Uo + h_prev @ self.Wo + self.bo)
        candt = tanh(xt @ self.Ug + h_prev @ self.Wg + self.bg)

        ct = ft * c_prev + it * candt
        ht = tanh(ct) * ot
        return ht, ct, candt, ot, it, ft
    
    def forward(self, x):
        T = x.shape[0]

        self.H = np.zeros((T+1, 1, self.Wf.shape[1]))
        self.C = np.zeros((T+1, 1, self.Wf.shape[1]))

        Y = np.zeros((T, self.Wy.shape[1]))
        self.O = np.zeros((T, 1, self.Wf.shape[1]))
        self.I = np.zeros((T, 1, self.Wf.shape[1]))
        self.F = np.zeros((T, 1, self.Wf.shape[1]))
        self.Cand = np.zeros((T, 1, self.Wf.shape[1]))

        for t in range(T):
            (self.H[t+1], self.C[t+1], self.Cand[t],
              self.O[t], self.I[t], self.F[t]) = self.forward_cell(x[t:t+1], self.H[t], self.C[t])
            Y[t] = self.last_activation(self.H[t+1] @ self.Wy + self.by)

        return Y
    
    def backward_cell(self, xt, h_prev, c_prev, c_current, candt, ot, it_gate, ft_gate, dh, dc_next):
        d_ot = dh * tanh(c_current) # Can be simplified.
        d_tanh_c = dh * ot
        d_c_from_h = d_tanh_c * tanh(c_current, derv=True) # Can be simplified.

        dct = d_c_from_h + dc_next
        
        d_ft = dct * c_prev
        d_it = dct * candt
        d_candt = dct * it_gate

        d_ft_pre = d_ft * derv_sigmoid(ft_gate)
        d_it_pre = d_it * derv_sigmoid(it_gate)
        d_ot_pre = d_ot * derv_sigmoid(ot)
        d_candt_pre = d_candt * derv_tanh(candt)

        dUft = xt.T @ d_ft_pre
        dWft = h_prev.T @ d_ft_pre
        dbft = np.sum(d_ft_pre, axis=0, keepdims=True)

        dUit = xt.T @ d_it_pre
        dWit = h_prev.T @ d_it_pre
        dbit = np.sum(d_it_pre, axis=0, keepdims=True)

        dUot = xt.T @ d_ot_pre
        dWot = h_prev.T @ d_ot_pre
        dbot = np.sum(d_ot_pre, axis=0, keepdims=True)

        dUgt = xt.T @ d_candt_pre
        dWgt = h_prev.T @ d_candt_pre
        dbgt = np.sum(d_candt_pre, axis=0, keepdims=True)

        dx_t = (d_ft_pre @ self.Uf.T +
                    d_it_pre @ self.Ui.T +
                    d_ot_pre @ self.Uo.T +
                    d_candt_pre @ self.Ug.T)

        dh_prev = (d_ft_pre @ self.Wf.T +
                    d_it_pre @ self.Wi.T +
                    d_ot_pre @ self.Wo.T +
                    d_candt_pre @ self.Wg.T)
        
        dc_prev = dct * ft_gate

        return (dUft, dWft, dbft,
                dUit, dWit, dbit,
                dUot, dWot, dbot,
                dUgt, dWgt, dbgt,
                dh_prev, dc_prev, dx_t)

    def _init_grads(self, U, W, b):
        return np.zeros_like(U), np.zeros_like(W), np.zeros_like(b)
    
    def backward(self, x, y_true, y_pred, learn=True):
        T = x.shape[0]

        dUf, dWf, dbf = self._init_grads(self.Uf, self.Wf, self.bf)
        dUi, dWi, dbi = self._init_grads(self.Ui, self.Wi, self.bi)
        dUo, dWo, dbo = self._init_grads(self.Uo, self.Wo, self.bo)
        dUg, dWg, dbg = self._init_grads(self.Ug, self.Wg, self.bg)

        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros((1, self.Wf.shape[1]))
        dc_next = np.zeros((1, self.Wf.shape[1]))
        dx = np.zeros_like(x)

        for t in reversed(range(T)):
            xt = x[t:t+1]
            yt = y_true[t:t+1]
            outp = y_pred[t:t+1]
            h_prev = self.H[t]
            h_t = self.H[t+1]
            c_prev = self.C[t]
            c_t = self.C[t+1]
            candt = self.Cand[t]
            ot = self.O[t]
            it_gate = self.I[t]
            ft_gate = self.F[t]

            dL = self.loss(yt, outp, derv=True)

            dWy += h_t.T @ dL
            dby += np.sum(dL, axis=0, keepdims=True)

            dh = dL @ self.Wy.T * self.last_activation(h_t, derv=True) + dh_next

            (dUf_t, dWf_t, dbf_t,
            dUi_t, dWi_t, dbi_t,
            dUo_t, dWo_t, dbo_t,
            dUg_t, dWg_t, dbg_t,
            dh_next, dc_next, dx[t]) = self.backward_cell(xt, h_prev, c_prev, c_t, candt, ot, it_gate, ft_gate, dh, dc_next)

            dUf += dUf_t; dWf += dWf_t; dbf += dbf_t
            dUi += dUi_t; dWi += dWi_t; dbi += dbi_t
            dUo += dUo_t; dWo += dWo_t; dbo += dbo_t
            dUg += dUg_t; dWg += dWg_t; dbg += dbg_t

        dUf /= T; dWf /= T; dbf /= T
        dUi /= T; dWi /= T; dbi /= T
        dUo /= T; dWo /= T; dbo /= T
        dUg /= T; dWg /= T; dbg /= T
        dWy /= T; dby /= T

        grads = (dUf, dWf, dbf,
                 dUi, dWi, dbi,
                 dUo, dWo, dbo,
                 dUg, dWg, dbg,
                 dWy, dby)
        
        if learn:
            self.update_params(grads)

        return dx


    def train(self, x, y, epochs=100, batch_size=8, print_every=0.1):
        losses = []
        for epoch in range(1, epochs+1):
            for batch in range(0, x.shape[0], batch_size):
                x_batch = x[batch:batch+batch_size]
                y_batch = y[batch:batch+batch_size]
                y_pred = self.forward(x_batch)

                self.backward(x_batch, y_batch, y_pred, learn=True)
            
            loss = self.loss(y, self.forward(x))
            losses.append(loss)
            if epoch % max(1, int(epochs*print_every)) == 0:
                print(f'Epoch: [{epoch}/{epochs}]> Loss: {loss}')
        return losses

class AutoEncoder(Model):
    count = 0
    def __init__(self, encoder_neurons: list, encoder_activations: list, decoder_neurons: list, decoder_activations: list, name=None):
        if name:
            self.name = name
        else:
            self.name = f'{self.__class__.__name__}_{__class__.count}'
        self.encoder = DNN(encoder_neurons, encoder_activations, f'{self.name}_Encoder')
        self.decoder = DNN(decoder_neurons, decoder_activations, f'{self.name}_Decoder')
        self.compiled = False
        __class__.count += 1

    def compile(self, loss, optimizer, accuracy):
        self.encoder.compile(loss, optimizer, accuracy)
        self.decoder.compile(loss, optimizer.copy(), accuracy)
        self.compiled = True

    def forward(self, x):
        latent = self.encoder.forward(x)
        outp = self.decoder.forward(latent)
        return outp, latent
    
    def backward(self, y=None, outp=None, dout=None, learn=True):
        dout = self.decoder.backward(y, outp, learn=learn)
        dout = self.encoder.backward(dout=dout, learn=learn)
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

                predictions, _ = self.forward(x_batch)
                total_loss += self.encoder.loss(y_batch, predictions)
                total_acc += self.encoder.accuracy(y_batch, predictions)

                self.backward(y_batch, predictions, learn=True)
                print(f'Epoch: {ep} | Batch: {idx}/{num_batches}', end='\r')
            
            avg_loss = total_loss / (x.shape[0] if self.encoder.loss.mode == 'sum' else num_batches)
            avg_acc = total_acc / (x.shape[0] if self.encoder.accuracy.mode == 'sum' else num_batches)
            
            if verbose and ep % print_every == 0:
                message = f'Epoch: {ep}, Loss: {avg_loss}, Acc: {avg_acc}'
                if x_test is not None and y_test is not None:
                    
                    total_loss = 0.0
                    total_acc = 0.0

                    for b in range(0, x_test.shape[0], batch_size):
                        x_batch = x_test[b:b+batch_size]
                        y_batch = y_test[b:b+batch_size]

                        predictions, _ = self.forward(x_batch)
                        total_loss += self.encoder.loss(y_batch, predictions)
                        total_acc += self.encoder.accuracy(y_batch, predictions)
                    avg_loss = total_loss / (x_test.shape[0] if self.encoder.loss.mode == 'sum' else num_batches)
                    avg_acc = total_acc / (x_test.shape[0] if self.encoder.accuracy.mode == 'sum' else num_batches)

                    message += f', Test Loss: {avg_loss}, Test Acc: {avg_acc}'
                print(message)

class GAN(Model):
    count = 0
    def __init__(self, generator_neurons: list, generator_activations: list, discriminator_neurons: list, discriminator_activations: list, name=None):
        if name:
            self.name = name
        else:
            self.name = f'{self.__class__.__name__}_{__class__.count}'
        self.generator = DNN(generator_neurons, generator_activations, name=f'{self.name}_Generator')
        self.discriminator = DNN(discriminator_neurons, discriminator_activations, name=f'{self.name}_Discriminator')
        self.compiled = False
        self.alr_epochs = 0
        __class__.count += 1


    def compile(self, loss, optimizer_generator, optimizer_discriminator):
        self.generator.compile(loss, optimizer_generator, None)
        self.discriminator.compile(loss, optimizer_discriminator, None)
        self.compiled = True
    
    def generate_examples(self, amount=1):
        noise = np.random.standard_normal(size=(amount, self.generator.W[0].shape[0]))
        examples = self.generator.forward(noise)
        return examples, noise
    
    def monitorize(self, imgs_path, num_examples):
        self.imgs_path = imgs_path
        self.seed = np.random.standard_normal(size=(num_examples, self.generator.W[0].shape[0]))

    def generate_and_save_images(self, epoch, gen_loss, disc_loss):
        clear_output(wait=True)
        generated_imgs = self.generator.forward(self.seed)
        if not os.path.exists(self.imgs_path): os.makedirs(self.imgs_path)
        rows = cols = int(self.seed.shape[0]**0.5)
        if rows * cols < self.seed.shape[0]:
            cols += 1
        fig, axes = plt.subplots(rows, cols, figsize=(6, 6))
        fig.suptitle(f'Epoch: {epoch}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}', fontsize=12)
        for i, ax in enumerate(axes.flat):
            if i < self.seed.shape[0]:
                ax.imshow(generated_imgs[i].reshape(28, 28), cmap='gray')
            ax.axis('off')
        plt.savefig(os.path.join(self.imgs_path, f'generated_images_epoch_{epoch}.png'))
        plt.show()

    def save_as_gif(self):
        pattern = os.path.join(self.imgs_path, 'generated_images_epoch_*.png')
        img_files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))

        images = [imageio.imread(img) for img in img_files]
        gif_filename = os.path.join(self.imgs_path, 'training.gif')
        imageio.mimsave(gif_filename, images, duration=0.5)
        print("GIF guardado como:", gif_filename)

    def train(self, x=None, epochs=1, batch_size=8, verbose=True):
        if not self.compiled: raise ValueError(f'model {self.name} is not compiled.')
        half_batch = batch_size // 2
        half_true = np.ones((half_batch, 1))
        half_false = np.zeros((half_batch, 1))
        true_labels = np.ones((batch_size, 1))
        labels = np.concatenate([half_true, half_false], axis=0)

        if x.shape[0] % half_batch != 0:
            x = x[:x.shape[0]-x.shape[0]%half_batch]

        for epoch in range(1, epochs+1):
            start = time.time()
            for batch in range(0, x.shape[0], half_batch):
                x_batch = x[batch:batch+half_batch]
                noise = np.random.standard_normal(size=(half_batch, self.generator.W[0].shape[0]))
                generated = self.generator.forward(noise)

                data = np.concatenate([x_batch, generated], axis=0)
                preds = self.discriminator.forward(data)
                disc_loss = self.discriminator.loss(labels, preds)
                self.discriminator.backward(labels, preds, learn=True) # Cambiar

                noise = np.random.standard_normal(size=(batch_size, self.generator.W[0].shape[0]))
                generated = self.generator.forward(noise)
                preds = self.discriminator.forward(generated)
                gen_loss = self.discriminator.loss(true_labels, preds)
                deltas = self.discriminator.backward(true_labels, preds, learn=False) # Cambiar
                self.generator.backward(dout=deltas, learn=True) # Cambiar

            if verbose:
                end = time.time()
                self.generate_and_save_images(epoch+self.alr_epochs, gen_loss, disc_loss)
                print(f'[{self.name}]> Epoch: ({epoch}/{epochs}) (in {end-start:.4f} seconds) G_loss: {gen_loss} / D_loss:{disc_loss}')

        self.alr_epochs += epochs