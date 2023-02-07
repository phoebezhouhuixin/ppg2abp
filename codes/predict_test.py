"""
    Computes the outputs for test data
"""

from helper_functions import *
from models import UNetDS64, MultiResUNet1D
import os
import tensorflow as tf
import time

def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

def main():
    print(os.getcwd())  
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
    # print(f"GPUs: {get_available_gpus()}") 
    # Loaded runtime CuDNN library: 7.4.1 but source was compiled with: 7.6.0. CuDNN library major and minor version needs to match. Upgrade your CuDNN library. 
    
    length = 1024               # length of signal

    dt = pickle.load(open(os.path.join('data','test.p'),'rb'))      # loading test data
    X_test = dt['X_test']
    Y_test = dt['Y_test']   
    print(f"Loaded PPG and ABP: {type(X_test)} {X_test.shape} and {Y_test.shape}") # (27260, 1024, 1), (27260, 1024, 1) 
    mdl1 = UNetDS64(length)                                             # creating approximation network
    print("Loading weights into ApproximateNetwork...")
    mdl1.load_weights(os.path.join('models','ApproximateNetwork.h5'))   # loading weights
    print("Loaded weights into ApproximateNetwork") 

    mdl2 = MultiResUNet1D(length)                                       # creating refinement network
    print("Loading weights into RefinementNetwork") 
    mdl2.load_weights(os.path.join('models','RefinementNetwork.h5'))    # loading weights
    print("Loaded weights into ApproximateNetwork") 

    start = time.time()
    Y_test_pred_approximate = mdl1.predict(X_test,verbose=1)            # predicting approximate abp waveform
    print(f"Approximation prediction complete {type(Y_test_pred_approximate[0])} {Y_test_pred_approximate[0].shape}")
    pickle.dump(Y_test_pred_approximate,open('test_output_approximate.p','wb')) # saving the approxmiate predictions
    
    Y_test_pred = mdl2.predict(Y_test_pred_approximate[0],verbose=1)    # predicting abp waveform
    print(f"Refined prediction complete {Y_test_pred.shape}")
    pickle.dump(Y_test_pred,open('test_output.p','wb'))                 # saving the predicted abp waeforms
    end = time.time()
    print(f"Duration: {end-start}")

if __name__ == '__main__':
    main()