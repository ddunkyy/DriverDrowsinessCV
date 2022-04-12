# DriverDrowsinessCV
DA 401 Capstone Project

Dang Pham

**Data**

Yawn eye dataset: https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new

UTA-RLDD: https://sites.google.com/view/utarldd/home

**Code**

There will be two main files for this project. The first is "model.py", this is the file contains structure of the neural network model. When this file runs, it will gather the images from the dataset and feed it into the model created. The training will take around 30 minutes for 100 epochs. This number can be changed and it will affect the accuracy of the model and computational time.

The second file is called "drowsiness detection.py". This file contains the code to run OpenCV for real-time detection. It should returns a separate window that open the webcam and start detecting face and determine drowsiness. This file is complicated to run because it utilizes tensorflow for GPU and set this up is troublesome. 