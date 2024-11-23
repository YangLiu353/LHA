# Lightweight Hybrid Attention RGB-D Networks for Accurate Camouflaged Object Detection

Source code and dataset for our paper "Lightweight Hybrid Attention RGB-D Networks for Accurate Camouflaged Object Detection" by Yang Liu, Shuhan Chen, Haonan Tang, and Shiyu Wang. 
The Visual Computer is under review.

# Training/Testing

The training and testing experiments are conducted using [PyTorch]( https://github.com/pytorch/ ) with one NVIDIA 3090 GPU of 24 GB Memory.

1.  Configuring your environment (Prerequisites):

  * Python 3.7+, Pytorch 1.5.0+, Cuda 10.2+, TensorboardX 2.1, opencv-python <br>
      If anything goes wrong with the environment, please check requirements.txt for details.

2.  Downloading necessary data:

  * Downloading dataset can be found from [Baidu Drive]( https://pan.baidu.com/s/1sTspmcOwoHyIwLP7CyQMmQ 提取码: 353X ).

# Evaluation
                                                                          
1.  CODToolbox：（ https://github.com/DengPingFan/CODToolbox ）- By DengPingFan(<https://github.com/DengPingFan>)

2.  Precision_and_recall：（ https://en.wikipedia.org/wiki/Precision_and_recall ）  
 
3.  BDE_Measure： [Baidu Drive]（ https://pan.baidu.com/s/1OorwYFq0ZY2I99ONoIF6Ig 提取码: 353X ）

# Network hyperparameters:

The epoch size and batch size are set to 100 and 10, respectively.
The PyTorch library was used to implement and train our model, which was trained using Adam optimization,
regularization was conducted using a weight decay of 1e-3, and we set the learning rate of our training phase to 1e-4.

# Reproduce

1.  Network training

  * python train.py   

  * parser.add_argument('--train_root', type=str, default='', help='the train images root')
   
    parser.add_argument('--val_root', type=str, default='', help='the val images root')
    
    parser.add_argument('--save_path', type=str, default='/', help='the path to save models and logs')

2.  Network testing

  * python test.py   

  * parser.add_argument('--test_path',type=str,default='/',help='test dataset path')

  * model.load_state_dict(torch.load('/Net_epoch_best.pth', weights_only=True))
   
    model_path = '/Net_epoch_best.pth'

  * save_path = '/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_path = '/' + dataset + '/'
    if not os.path.exists(edge_save_path):
        os.makedirs(edge_save_path)

3.   Evaluation of quantitative indicators

 *   python CODtest_metrics.py
    
 *   BDE_Measure: Need to run main.m in the file on Matlab

#  Architecture and Details

![fdb42aff9cac95a9c767dcebdbbf074](https://github.com/user-attachments/assets/69ee4c67-fc25-4469-a75a-f55e0a5df6d2)


# Results

![a235ce2b53954fa8fbed9d09c439342](https://github.com/user-attachments/assets/b2f7d501-7902-45a8-a540-fa2f260a64fa)


![bf14b4846cf7e103323a7c9f8206127](https://github.com/user-attachments/assets/344b8ce6-f7d8-40af-90a6-8309982d96d4)


