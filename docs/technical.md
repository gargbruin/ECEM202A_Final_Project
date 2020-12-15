# Table of Contents

1. [System Overview](#system-overview)
2. [Training Data](#training-data)  
   2.1 [Datasets Used](#datasets-used)  
   2.2 [Data Preprocessing](#data-preprocessing)
3. [Real-Time Data](#real-time-data)  
   3.1 [Data Collection](#data-collection)  
   3.2 [Data Preprocessing](#data-preprocessing)
4. [Machine Learning Model](#machine-learning-model)  
   4.1 [Model Topology](#model-topology)  
   4.2 [Model Training](#model-training)
5. [Real-Time Inference](#real-time-inference)  
6. [Platforms Used](#platforms-used)  
   6.1 [Google Drive](#google-drive)  
   6.2 [Google Colaboratory](#google-colaboratory)  
   6.3 [TensorFlow](#tensorflow)  
   
# System Overview 
In this project, we are using the sensor data retrieved from wearable devices to recognize and log the human activities using deep learning approach. We are using the IMU data (accelerometer, gyroscope) obtained from an Apple Watch worn by the user on their dominant hand. The activities recognized are logged and stored in a pdf document. The diagram below shows the overall approach used in this project:

<p align="center">
  <img src="png/technical_approach.png">  
</p>

We are targeting the detection of following human activities:
* Walking
* Sitting
* Eating
* Brushing Teeth

# Training Data

## Datasets Used

We started our project with PAMAP2 dataset but the trained model's accuracy was not good enough on the real-time data we collected. This was because PAMAP2 dataset is too small to train a large model successfully. Therefore, we tried with a bigger dataset (WISDM dataset) which has data for 51 users. This dataset was good enough to train a large model but the problem with this dataset is that after training, the model is used to predict activities from data collected in real-time which uses different sensors as compared to this dataset. So, to prevent the trained model from getting biased towards any particular dataset (i.e data collected from a particular type of sensor), we finally used a combination of WISDM and PAMAP2 datasets.

Link to the datasets: **[PAMAP2](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+) [WISDM](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)** 

Raw dat files for PAMAP2 can be found here : **[Data/PAMAP2_Dataset/Protocol](https://github.com/gargbruin/WALG/tree/main/Data/PAMAP2_Dataset/Protocol)**  
Raw csv files for WISDM can be found here : **[Data/WISDM_Dataset/raw/watch](https://github.com/gargbruin/WALG/tree/main/Data/WISDM_Dataset/raw/watch)**  

## Data Preprocessing

<p align="center">
  <img src="png/Training_Data_Preprocessing.png">  
</p>

Following are the various steps used in preprocessing the training datasets:
* **Sampling Frequency**: WISDM and PAMAP2 use sampling frequencies of 20 Hz and 100 Hz respectively. Therefore, to match their sampling frequencies, PAMAP2 dataset has been downsampled to match WISDM's frequency.
* **Sliding Window**: We are using the sliding window approach to split data into fixed size windows. With experimentation, we found that a sliding window of 10 seconds (200 samples) and a stride of 2 seconds (40 samples) gives us the best training and inference accuracy for all targeted activities. 
* **Mean and Standard Deviation**: Most of the reference HAR work uses multiple sensors at different locations (like ankles, wrist, chest etc). However, in this project, we are detecting human activities based on two sensors' data (accelerometer, gyroscope) only and that too from just one location (wrist). To recover the accuracy loss because of a limited number of input features (sensors), we added additional features like mean and standard deviation to the sampled windows.

Link to notebook used for PAMAP2 dataset preprocessing: **[Notebooks/Preprocess_PAMAP2.ipynb](https://github.com/gargbruin/WALG/blob/main/Notebooks/Preprocess_PAMAP2.ipynb)**  
Link to notebook used for WISDM+PAMAP2 dataset preprocessing: **[Notebooks/Preprocess_WISDM_PAMAP2.ipynb](https://github.com/gargbruin/WALG/blob/main/Notebooks/Preprocess_WISDM_PAMAP2.ipynb)**  
Processed numpy files for PAMAP2 can be found here : **[Data/PAMAP2](https://github.com/gargbruin/WALG/tree/main/Data/PAMAP2)**  
Processed numpy files for PAMAP2+WISDM can be found here : **[Data/WISDM_PAMAP2](https://github.com/gargbruin/WALG/tree/main/Data/WISDM_PAMAP2)**  

# Real-Time Data 

## Data Collection
We are using the SensorLog app on Apple Watch to collect 6-DOF IMU data (accelerometer and gyroscope). The Apple Watch is worn by the user on their dominant hand. The SensorLog app can sample the data at a frequency of upto 100 Hz. Since our processed training data has a sampling frequency of 20 Hz, we are using the app to collect the IMU data at the same frequency. The app provides us data in the csv format. Once we have the csv files from the Apple Watch, these files are uploaded on Google Drive for further processing and inference. 

<p align="center">
  <img src="png/MotionSenseHRV.png">  
  <img src="png/Apple_Watch_SensorLog.png">  
</p>

Raw csv files can be found here : **[Data/Live_Data/raw](https://github.com/gargbruin/WALG/tree/main/Data/Live_Data/raw)**  

## Data Processing

<p align="center">
  <img src="png/Real_Time_Data_Preprocessing.png">  
</p>

We run a preprocessing python script on the raw data that performs the following operations:
* **Sampling Frequency**: SensorLog app allows us to directly sample  the data at 20 Hz.
* **Sliding Window**: We are using the sliding window approach to split data into fixed size windows of 10 seconds (200 samples) and a stride of 2 seconds (40 samples). 
* **Mean and Standard Deviation**: Similar to the training dataset, we are adding additional features like mean and standard deviation to the sampled windows.

Link to notebook used for real-time data preprocessing: **[Notebooks/Preprocess_Live_Data.ipynb](https://github.com/gargbruin/WALG/blob/main/Notebooks/Preprocess_Live_Data.ipynb)**  
Processed numpy files can be found here : **[Data/Live_Data/processed](https://github.com/gargbruin/WALG/tree/main/Data/Live_Data/processed)**

# Machine Learning Model

## Model Topology

We experimented with multiple different neural network topologies like MLPs, CNNs, LSTMs and ConvLSTMs and found that CNNs work the best for us.
To improve the model’s accuracy, we tried many different network configurations (like changing number of layers, number of features per layer, adding different types of layers like BatchNormalization, Dropout). Following is the final network topology we are using for making predictions on real-time data:

<p align="center">
  <img src="png/Model_Topology.png">  
</p>

## Model Training

Our combined training data has 60 unique users which performed all the targeted activities. To perform validation and testing accurately, users to validate and test the model are not being used for training. The processed training dataset has been divided into following subsets:

<p align="center">
  <img src="png/Data_Splitting.png">  
</p>

We tried training the model with different number of epochs and batch sizes and settled with 50 epochs with a batch size of 256. It takes around 3 minutes and 20 seconds to train and validate the model. After training, the best model is saved which is later on loaded for inference with the test data and real-time data.

Link to notebook used for network training: **[Notebooks/WALG.ipynb](https://github.com/gargbruin/WALG/blob/main/Notebooks/WALG.ipynb)**

# Real-Time Inference

# Platforms Used

<p align="center">
  <img src="png/Platforms.png">  
</p>

## Google Drive

We stored all our data on Google Drive. The Shared Drives feature helped us a lot in collaboration.

## Google Colaboratory

All python scripting was done on Google Colab. Linking Drive with Colab allowed easy access to the data.

## TensorFlow

All the model development and training was done using TensorFlow.

