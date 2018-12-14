import tensorflow as tf
import numpy as np
import os
if not os.path.exists('model'):
    os.makedirs('model')
if not os.path.exists('img_out'):
    os.makedirs('img_out')
from PIL import Image
import ground_truth_label as gtl
from matplotlib import pyplot as plt
import gc

num_label  = 32


""" Some funtions """
def max_pool(X):
    # Works only for GPU
    H , Mask = tf.nn.max_pool_with_argmax(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    Mask = tf.stop_gradient(Mask)
    return H, Mask


def unpool(X_mp , Mask ,batch_size, pad1 , pad2):
   [_, im_1, im_2, ch] = X_mp.get_shape().as_list()
   
   Batch_size = batch_size
   #Turn values into 1D
   X_mp = tf.reshape(X_mp,[-1])
   #Prepare 2nd entries of the indices coord
   Mask = tf.reshape(Mask,[-1, im_1*im_2*ch,1])
   #Prepare 1st entries of the indices coord
   Trial_idx = tf.range(Batch_size,dtype=tf.int64)
   Trial_idx = tf.reshape(Trial_idx,[-1,1,1])
   Trial_idx = tf.tile(Trial_idx,[1,im_1*im_2*ch,1])
   #Combine the two entries
   # Mask will be the coords set len(X_mp) * 2
   Mask = tf.concat([Trial_idx,Mask],axis=-1)
   Mask = tf.reshape(Mask,[-1,2])
   #Prepare output dims
   if pad1 == False:
       imo_1 = im_1*2
   else:
       imo_1 = im_1*2-1
       
   if pad2 == False:
       imo_2 = im_2*2
   else:
       imo_2 = im_2*2-1  
   Out_len = tf.constant([Batch_size,imo_1*imo_2*ch],dtype=tf.int64)
   #Feed in the indices and values with output size
   X_unpool = tf.scatter_nd(Mask,X_mp,Out_len)
   X_unpool = tf.reshape(X_unpool,[-1,imo_1,imo_2,ch])
   return X_unpool


def get_batch_id(batch_size,datalen):
    id_all = np.arange(datalen)
    np.random.shuffle(id_all)   
    id_list = []    
    for i in range(int(datalen/batch_size)):
        id_batch = id_all[int(i*batch_size):int(i*batch_size)+batch_size]
        id_list.append(id_batch)        
    if datalen % batch_size !=0:
        i+=1
        id_batch = id_all[int(i*batch_size):]
        id_list.append(id_batch)        
    return id_list

    

def conv_en(H,filters,layer_name,load_pretrain,param_dict):
    if load_pretrain:
        initK_en  = tf.constant_initializer(param_dict[layer_name][0])
        initb_en  = tf.constant_initializer(param_dict[layer_name][1])
    else:
        initK_en = tf.contrib.layers.xavier_initializer()
        initb_en = tf.zeros_initializer()        
    return  tf.layers.conv2d(H,filters=filters,kernel_size=(3,3),strides=(1, 1),padding='SAME',
                         activation=None ,kernel_initializer=initK_en,bias_initializer=initb_en)

def conv_de(H,filters,initK_de):
    return tf.layers.conv2d(H,filters=filters,kernel_size=(3,3),strides=(1, 1),padding='SAME',
                                 activation=None ,kernel_initializer=initK_de)  

""" Define Segnet """
class segnet():
    def __init__(self,batch_size):

        tf.reset_default_graph()
        self.gf = tf.Graph()
        self.batch_size    = batch_size
        
    def build_gf(self,load_pretrain=False):
        with self.gf.as_default():
            if load_pretrain:
                param_dict = np.load('vgg16.npy', encoding='latin1').item()
                print("Use pretrained")
            else:
                param_dict = {}
                print("Without pretrained")
            initK_de  = tf.contrib.layers.xavier_initializer()
            #Placeholders
            self.X  = tf.placeholder(tf.float32, [None, 180,240, 3])
            self.Y  = tf.placeholder(tf.float32, [None, 180,240,32])
            self.T  = tf.placeholder(tf.bool)
            self.LR = tf.placeholder(tf.float32)
        
            ## Encoder ##
            H = conv_en(self.X,filters=64,layer_name='conv1_1',load_pretrain=load_pretrain,param_dict=param_dict)
            H          = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H          = tf.nn.relu(H)
            H = conv_en(H,filters=64,layer_name='conv1_2',load_pretrain=load_pretrain,param_dict=param_dict)
            H          = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H          = tf.nn.relu(H)
            H, mask1 = max_pool(H) 
            # 90, 120,  64                                                             
            
             
            H = conv_en(H,filters=128,layer_name='conv2_1',load_pretrain=load_pretrain,param_dict=param_dict)               
            H = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H          = tf.nn.relu(H)
            H = conv_en(H,filters=128,layer_name='conv2_2',load_pretrain=load_pretrain,param_dict=param_dict)                
            H = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H          = tf.nn.relu(H)
            H, mask2 = max_pool(H)       
            # 45,  60,  128                                                      
        
            
            H = conv_en(H,filters=256,layer_name='conv3_1',load_pretrain=load_pretrain,param_dict=param_dict)                                 
            H = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H          = tf.nn.relu(H)
            H = conv_en(H,filters=256,layer_name='conv3_2',load_pretrain=load_pretrain,param_dict=param_dict)                                  
            H = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H          = tf.nn.relu(H)
            H, mask3 = max_pool(H)          
            # 23,  30, 256                                                     
        
        
            ## Decoder ##
            H = unpool(H,mask3, self.batch_size , pad1=True , pad2=False)
            H = conv_de(H,filters=256,initK_de=initK_de)
            H = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H = tf.nn.relu(H)
            H = conv_de(H,filters=128,initK_de=initK_de)
            H = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H = tf.nn.relu(H)
            # 45,  60, 128 
            
            
            H = unpool(H,mask2, self.batch_size , pad1=False , pad2=False)      
            H = conv_de(H,filters=128,initK_de=initK_de)
            H = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H = tf.nn.relu(H)
            H = conv_de(H,filters=64,initK_de=initK_de)                    
            H = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H = tf.nn.relu(H)
            # 90, 120,  64
            
            
            H = unpool(H,mask1, self.batch_size , pad1=False , pad2=False) 
            H = conv_de(H,filters=64,initK_de=initK_de)                                       
            H = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H = tf.nn.relu(H)
            H = conv_de(H,filters=num_label,initK_de=initK_de)                                            
            H = tf.layers.batch_normalization(H,momentum=0.1,training=self.T)
            H = tf.nn.relu(H)
            # 180,  240, num_label
            
            self.Y_pred = tf.nn.softmax(H,axis=-1)
            self.Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Y_pred, labels=self.Y))

            Y_correct_test   = tf.equal(tf.argmax(self.Y_pred,3), tf.argmax(self.Y,3))
            Y_correct_test   = tf.cast(Y_correct_test, tf.float32)
            self.Acc         = tf.reduce_mean(Y_correct_test)    

            Update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(Update_ops):
                self.Train_step = tf.train.AdamOptimizer(self.LR).minimize(self.Loss)
                
            print("Graph Loaded")
            
            
    def train(self,x_train,y_train,x_test,y_test,lr,num_epoch,save_se=False,save_fig=True,print_fig=True,print_freg=10):
        
        train_len = len(y_train)

        with tf.Session(graph=self.gf) as se:
            se.run(tf.global_variables_initializer())
            if save_se:
#                ListV = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=None)
#                print(ListV)
#                saver = tf.train.Saver(var_list=ListV,max_to_keep=1)
                saver = tf.train.Saver(max_to_keep=0)
                print("Session will be saved")
            else:
                print("Session won't be saved")
                
            for epoch in range(num_epoch):
                id_list = get_batch_id(self.batch_size,train_len)
                for batch_id in id_list:
                    batch_x = []
                    batch_y = []
                    for idx in (batch_id):
                        batch_x.append(x_train[idx])
                        batch_y.append(y_train[idx])

                    _ , loss_i , acc_train =  se.run([self.Train_step,self.Loss,self.Acc] , 
                                feed_dict={self.X: batch_x, self.Y: batch_y, 
                                           self.T: True, self.LR:lr})
        
                loss_test , acc_test , y_test_pred =  se.run([self.Loss,self.Acc,self.Y_pred] , 
                            feed_dict={self.X: x_test, self.Y: y_test, self.T: False})
                    

                if epoch<5 or (epoch+1)%print_freg==0:
                    # gc.collect()
                    img_out     = np.argmax(y_test_pred,3)[0]
                    img_out_rgb = gtl.label2rgb(img_out)
                    if save_fig:
                        image = Image.fromarray(img_out_rgb.astype('uint8'), 'RGB')
                        image.save("img_out/epoch_" + str(epoch) + ".png", "png")
                    if print_fig:
                        plt.imshow(img_out_rgb)
                        plt.grid(False)
                        plt.axis('off')
                        plt.show()
    
                print('Epoch:', epoch, '| test: %.4f' % acc_test, '| train: %.4f' % acc_train,
                      '| train Loss: %.4f' % loss_i, '| test Loss: %.4f' % loss_test)    
            if save_se:
                print("Saving session")
                saver.save(se, "model/net1.ckpt")
                print("Saved")
        print("Training Completed")

        
    def test(self,x_test,y_test):
        with tf.Session(graph=self.gf) as se:
            se.run(tf.global_variables_initializer())
#            ListV = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=None)
#            saver = tf.train.Saver(var_list=ListV)
#            ckpt = tf.train.get_checkpoint_state('model')
#            saver.restore(se, ckpt.model_checkpoint_path)
            
            saver = tf.train.Saver()
            saver.restore(se,"model/net1.ckpt")
        
            loss_test , acc_test , y_test_pred =  se.run([self.Loss,self.Acc,self.Y_pred] , 
                    feed_dict={self.X: x_test, self.Y: y_test, 
                               self.T: False})
        
            img_out     = np.argmax(y_test_pred,3)[0]
            img_out_rgb = gtl.label2rgb(img_out)
            plt.imshow(img_out_rgb)
            plt.grid(False)
            plt.axis('off')
            plt.show()
        
            print('test acc: %.4f' % acc_test,'| test Loss: %.4f' % loss_test) 
            
        return img_out_rgb




