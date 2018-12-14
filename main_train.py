#import tensorflow as tf

""" Train """
import load_np as load
x_train,y_train,x_test,y_test = load.train_test_data()
batch_size = 5
lr=0.005
import sgnet_class as sgc
sg_net = sgc.segnet(batch_size)
sg_net.build_gf(load_pretrain=True)
sg_net.train(x_train,y_train,x_test,y_test,lr,num_epoch=10,save_se=True,save_fig=True,print_fig=False,print_freg=10)




