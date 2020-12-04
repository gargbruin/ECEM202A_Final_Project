# System Overview 
In this project, we are using the sensor data retrieved from wearable devices to recognize and log the human activities using deep learning approach. We are using the IMU data (accelerometer, gyroscope) obtained from an Apple Watch worn by the user on their dominant hand. The activities recognized are logged and stored in a pdf document. The diagram below shows the overall approach used in this project:

![System Diagram](png/technical_approach.png)

# Data Collection and Preprocessing
We are using the SensorLog app on Apple Watch to collect 6-DOF IMU data (accelerometer and gyroscope). The Apple Watch is worn by the user on their dominant hand. The SensorLog app can sample the data at a frequency of upto 100 Hz. Since our training data was collected at a sampling frequency of 20 Hz, we are using the app to collect the IMU data at the same frequency. The app provides us data in the csv format. Once we have the csv files from the Apple Watch, we upload them on Google Drive and run a preprocessing python script which peforms the following operations: 
    * Filter the raw data to extract relevant features 
    * Apply a sliding window to generate input data for the neural network 
    * Save the processed data as numpy files 
Raw csv files can be found here : data/WALG_inference/raw/*.csv
Processed numpy files can be found here : data/WALG_inference/processed/*.npy

# Algorithms
# Datasets
# Platform
