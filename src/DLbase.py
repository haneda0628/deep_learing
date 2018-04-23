import numpy as np
import copy

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

class AffineLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx

class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = copy.copy(x)
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

def test_addLayer():
    addLayer = MulLayer()
    v = addLayer.forward(3,4)
    print v
    dv = addLayer.backward(10)
    print dv

def test_ReLULayer():
    layer = ReLULayer()
    y = layer.forward([-3,4])
    print y
    dx = layer.backward([10,3])
    print dx

def test_AffineLayer():
    layer = AffineLayer(
    np.array(
    [
        [2,2,4],
        [3,4,2],
        [5,1,3]
    ]),
        np.array([3,4,2])

    )

    y = layer.forward(np.array([2,3,4]))
    print y
    dx = layer.backward(np.array([8,9,6]))
    print dx

test_AffineLayer()
