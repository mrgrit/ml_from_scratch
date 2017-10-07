import numpy as np

def sigmoid(x):
    # np.exp : just used for matrix input
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def identity_function(x):
    return x



def test(test_no):
    if test_no == 1:
        # implement neural network
        x = np.array([1.0,0.5])
        w1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
        b1 = np.array([0.1,0.2,0.3])

        print("shape of w1 ==> ", w1.shape)
        print("shape of b1 ==> ", b1.shape)
        print("shape of x ==> ", x.shape)

        a1 = np.dot(x,w1) + b1
        print("a1 ==> ", a1)

        print("relu ==> ", relu(a1))

        z1 = sigmoid(a1)
        print("sigmoid ==> ", z1)

    if test_no == 2:
        x = np.array([1.0,0.5])
        w1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
        b1 = np.array([0.1,0.2,0.3])
        a1 = np.dot(x,w1) + b1
        z1 = sigmoid(a1)

        w2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
        b2 = np.array([0.1,0.2])

        print("shape of z1 ==> ", z1.shape)
        print("shape of w2 ==> ", w2.shape)
        print("shape of b2 ==> ", b2.shape)

        a2 = np.dot(z1,w2) + b2
        print("layer2 output ==> ", a2)
        
        z2 = sigmoid(a2)
        print("sigmoid(a2) ==> ", z2)

    if test_no == 3:
        x = np.array([1.0,0.5])
        w1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
        b1 = np.array([0.1,0.2,0.3])
        a1 = np.dot(x,w1) + b1
        z1 = sigmoid(a1)

        w2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
        b2 = np.array([0.1,0.2])
        a2 = np.dot(z1,w2) + b2
        z2 = sigmoid(a2)

        w3 = np.array([[0.1,0.3],[0.2,0.4]])
        b3 = np.array([0.1,0.2])

        a3 = np.dot(z2,w3) + b3
        y = identity_function(a3) # or y = a3

        print("y ===> ",y)

    if test_no == 4:
        def init_network():
            network = {}
            network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
            network['b1'] = np.array([0.1,0.2,0.3])
            network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
            network['b2'] = np.array([0.1,0.2])
            network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
            network['b3'] = np.array([0.1,0.2])

            return network

        def forward(network, x):
            W1,W2,W3 = network['W1'],network['W2'],network['W3']
            b1,b2,b3 = network['b1'],network['b2'],network['b3']

            a1 = np.dot(x,W1) + b1
            z1 = sigmoid(a1)
            a2 = np.dot(z1,W2) + b2
            z2 = sigmoid(a2)
            a3 = np.dot(z2,W3) + b3
            y = identity_function(a3)

            return y

        network = init_network()
        x = np.array([1.0,0.5])
        y = forward(network,x)
        print("y ===> ", y)

test(4)
