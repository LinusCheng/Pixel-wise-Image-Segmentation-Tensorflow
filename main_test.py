#import numpy as np
#import tensorflow as tf
#from PIL import Image
#from matplotlib import pyplot as plt

""" Test """
import load_np as load
x_test,y_test = load.test_data()
batch_size = 5
import sgnet_class as sgc
sg_net = sgc.segnet(batch_size)
sg_net.build_gf()
img_out_rgb = sg_net.test(x_test,y_test)

