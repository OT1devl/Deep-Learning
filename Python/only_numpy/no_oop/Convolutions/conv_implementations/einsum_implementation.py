import numpy as np
import time

def im2col_strided(x, field_height, field_width, padding=1, stride=1):
    m, H, W, C = x.shape
    out_h = (H + 2 * padding - field_height) // stride + 1
    out_w = (W + 2 * padding - field_width) // stride + 1
    x_padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    shape = (m, out_h, out_w, field_height, field_width, C)
    strides = (
        x_padded.strides[0],
        stride * x_padded.strides[1],
        stride * x_padded.strides[2],
        x_padded.strides[1],
        x_padded.strides[2],
        x_padded.strides[3]
    )
    return np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides), out_h, out_w

def fast_convolution(x, W, b, padding=1, stride=1):
    m, H, W_in, C = x.shape
    fh, fw, _, K = W.shape
    x_strided, out_h, out_w = im2col_strided(x, fh, fw, padding, stride)

    out = np.einsum('mxyhwc,hwck->mxyk', x_strided, W, optimize='optimal')
    out += b.reshape(1, 1, 1, -1)
    return out.reshape(m, out_h, out_w, K)

def fast_convolution_backprop(x, W, b, dout, padding=1, stride=1):

    m, H, W_in, C = x.shape
    fh, fw, _, K = W.shape
    x_strided, out_h, out_w = im2col_strided(x, fh, fw, padding, stride)
    dout_reshaped = dout.reshape(m, out_h, out_w, K)
    
    dW = np.einsum('mxyhwc,mxyk->hwck', x_strided, dout_reshaped, optimize='optimal')
    
    db = np.sum(dout_reshaped, axis=(0, 1, 2)).reshape(1, -1)
    
    dx_strided = np.einsum('mxyk,hwck->mxyhwc', dout_reshaped, W, optimize='optimal')
    dx_padded = np.zeros((m, H + 2*padding, W_in + 2*padding, C), dtype=x.dtype)
    
    for h in range(fh):
        for w in range(fw):
            dx_padded[:, 
                      h: h + stride*out_h: stride,
                      w: w + stride*out_w: stride,
                      :] += dx_strided[:, :, :, h, w, :]

    return (dx_padded[:, padding:-padding, padding:-padding, :] if padding > 0 else dx_padded), dW, db

def test_convolution():
    np.random.seed(42)

    batch_size = 200
    input_size = 224
    channels_in = 3
    num_filters = 4
    filter_size = 3
    padding = 1
    stride = 1
    
    x = np.random.randn(batch_size, input_size, input_size, channels_in).astype(np.float32)
    W = np.random.randn(filter_size, filter_size, channels_in, num_filters).astype(np.float32)

    b = np.random.randn(1, num_filters).astype(np.float32)
    
    start = time.time()
    output = fast_convolution(x, W, b, padding, stride)
    total = time.time() - start
    print(f'Output Shape: {output.shape}, time: {total:.4f} seconds')
    

    start_2 = time.time()
    dout = np.random.randn(*output.shape).astype(np.float32)
    dx, dW, db = fast_convolution_backprop(x, W, b, dout, padding, stride)
    end_2 = time.time()
    print(f"Backward time: {end_2 - start_2:.4f} seconds")
    
    assert dx.shape == x.shape, f"Error in dx.shape: {dx.shape} != {x.shape}"
    assert dW.shape == W.shape, f"Error in dW.shape: {dW.shape} != {W.shape}"
    assert db.shape == (1, num_filters), f"Error in db.shape: {db.shape}"
    print(f"All tests passed! Total time: {total + (end_2 - start_2):.4f} seconds")

if __name__ == "__main__":
    test_convolution()