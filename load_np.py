import numpy as np
import os
if not os.path.exists('model'):
    os.makedirs('model')


def train_test_data():
    X_np = np.load('X_Y_CamSeq01.npz')['X_np']
    Y_np = np.load('X_Y_CamSeq01.npz')['Y_np']
    x_train = np.float32(X_np[0:95,:,:,:])
    y_train = np.float32(Y_np[0:95,:,:])
    x_test = np.float32(X_np[96:,:,:,:])
    y_test = Y_np[96:,:,:]

    x_train_ls = []
    y_train_ls = []
    x_test_ls  = []
    y_test_ls  = []

    for i in range(x_train.shape[0]):
        x_train_ls.append(x_train[i])
        y_train_ls.append(y_train[i])
    for i in range(x_test.shape[0]):
        x_test_ls.append(x_test[i])
        y_test_ls.append(y_test[i])
    return x_train_ls,y_train_ls,x_test_ls,y_test_ls


def test_data():
    X_np = np.load('X_Y_CamSeq01.npz')['X_np']
    Y_np = np.load('X_Y_CamSeq01.npz')['Y_np']
    x_test = np.float32(X_np[96:,:,:,:])
    y_test = Y_np[96:,:,:]
    x_test_ls  = []
    y_test_ls  = []
    for i in range(x_test.shape[0]):
        x_test_ls.append(x_test[i])
        y_test_ls.append(y_test[i])
    return x_test_ls,y_test_ls

#class load_data():
#    def __init__(self):
#        pass
#    def train_test_data(self):
#        X_np = np.load('X_Y_CamSeq01.npz')['X_np']
#        Y_np = np.load('X_Y_CamSeq01.npz')['Y_np']
#        x_train = np.float32(X_np[0:95,:,:,:])
#        y_train = np.float32(Y_np[0:95,:,:])
#        x_test = np.float32(X_np[96:,:,:,:])
#        y_test = Y_np[96:,:,:]
#        del X_np, Y_np
#        return x_train,y_train,x_test,y_test
#    def test_data(self):
#        X_np = np.load('X_Y_CamSeq01.npz')['X_np']
#        Y_np = np.load('X_Y_CamSeq01.npz')['Y_np']
#        x_test = np.float32(X_np[96:,:,:,:])
#        y_test = Y_np[96:,:,:]
#        del X_np, Y_np
#        return x_test,y_test