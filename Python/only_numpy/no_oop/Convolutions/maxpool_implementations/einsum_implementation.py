import numpy as np
import time

def im2col_strided(x, field_height, field_width, padding=0, stride=1):
    m, H, W, C = x.shape
    out_h = (H + 2 * padding - field_height) // stride + 1
    out_w = (W + 2 * padding - field_width) // stride + 1
    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        x_padded = x
    shape = (m, out_h, out_w, field_height, field_width, C)
    strides = (x_padded.strides[0],
               stride * x_padded.strides[1],
               stride * x_padded.strides[2],
               x_padded.strides[1],
               x_padded.strides[2],
               x_padded.strides[3])
    return np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides), out_h, out_w

def fast_maxpool(x, pool_height, pool_width, stride=1, padding=0):
    x_strided, out_h, out_w = im2col_strided(x, pool_height, pool_width, padding, stride)
    out = np.max(x_strided, axis=(3, 4))
    return out, x_strided

def fast_maxpool_backprop(x, pool_height, pool_width, stride, padding, dout, x_strided):
    m, out_h, out_w, ph, pw, C = x_strided.shape

    max_val = np.max(x_strided, axis=(3, 4), keepdims=True)
    mask = (x_strided == max_val)
    mask = mask / np.sum(mask, axis=(3, 4), keepdims=True)
    dout_expanded = dout[:, :, :, None, None, :]

    dpatch = mask * dout_expanded
    m, H, W, C = x.shape
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    dx_padded = np.zeros((m, H_padded, W_padded, C), dtype=x.dtype)

    for i in range(pool_height):
        for j in range(pool_width):
            dx_padded[:, i: i + stride * out_h: stride, j: j + stride * out_w: stride, :] += dpatch[:, :, :, i, j, :]
    
    if padding > 0:
        dx = dx_padded[:, padding:-padding, padding:-padding, :]
    else:
        dx = dx_padded
    return dx

def test_maxpool():
    np.random.seed(42)
    batch_size = 10
    input_size = 224
    channels = 3
    pool_size = 2
    stride = 2
    padding = 0
    x = np.random.randn(batch_size, input_size, input_size, channels).astype(np.float32)
    start = time.time()
    out, x_strided = fast_maxpool(x, pool_size, pool_size, stride, padding)
    total = time.time() - start
    print(f'Forward Maxpool: Output Shape: {out.shape}, time: {total:.4f} seconds')
    dout = np.random.randn(*out.shape).astype(np.float32)
    start_bp = time.time()
    dx = fast_maxpool_backprop(x, pool_size, pool_size, stride, padding, dout, x_strided)
    bp_time = time.time() - start_bp
    print(f"Backward Maxpool time: {bp_time:.4f} seconds")
    assert dx.shape == x.shape, f"Error en dx.shape: {dx.shape} != {x.shape}"
    print("Test Maxpool passed!")

if __name__ == "__main__":
    test_maxpool()
