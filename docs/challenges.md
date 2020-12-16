# Challenge 1

* Most of the reference HAR work uses multiple sensors at different locations.  
* It was challenging for us to get better accuracy using just Accelerometer and Gyroscope data from a wrist sensor.  
* To improve the accuracy, we enhanced the dataset by adding more features like mean and standard deviation.  

# Challenge 2

* We started our project using the MotionSense HRV wearable and the mCerebrum app to collect real-time data.  
* However, the app was giving a lot of problems. For example, we were seeing multiple data values for same timestamps in the csv file as shown in the following figure:

<p align="center">
  <img src="png/Challenge_2.png">  
</p>

* Debugging the MotionSense HRV wearable and app took around 2 weeks.  
* This left us with less time to work on the project using Apple Watch.  

# Challenge 3

* Model training and real-time inference are performed on data collected from different sensors.  
* To prevent the trained model from getting biased to data from one type of sensor, we trained the model on two different datasets (using different sensors).  
* We observed that using this combination of datasets improved our real-time inference accuracy significantly. 

# Challenge 4

* To train our model with good accuracy, we needed datasets with a large number of users and with our target sensors and activities.  
* Looking for such relevant datasets took a significant amount of time.
