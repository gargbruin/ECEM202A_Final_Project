# Overview

With the ubiquitous presence of smart wearable devices like Apple Watch, Fitbit bands etc around us, a huge amount of data is now available for data mining research and data mining applications. These devices have powerful sensors like Inertial Measurement Units (IMU), Photoplethysmogram (PPG), GPS and audio sensors which are capable of reading and processing the data quite accurately and in real time. A lot of classical or machine learning based techniques have been proposed to generate insights from the collected sensor data. 

One such area where this sensor data can be used is Human Activity Recognition (HAR) which refers to predicting what a person is doing from the series of the observation of person’s action.  This is an active area of research. HAR provides personalized support for various applications and is associated with a wide range of fields of study like medicinal services, dependable automation developing, and smart surveillance system. 

# Project Goals

The project is aimed to build a system which uses machine learning models to accurately detect and log the human activity using the sensor data (from the IMU embedded in the wearables). Such a system can have a large number of applications. There is a possibility of providing a highly personalised and customised experience of smart devices like phones and watches. Also it can be determined if the user is following a healthy exercise routine or not.

# How is HAR done today?
Human activity recognition has gained importance in recent years due to its applications in various fields such as health, security and surveillance, entertainment, and intelligent environments. A significant amount of work has been done on human activity recognition and researchers have leveraged different approaches, such as wearable, object-tagged, and device-free, to recognize human activities. Classical approaches to the problem involve hand crafting features from the time series data based on fixed-size windows and training machine learning models, such as ensembles of decision trees. The difficulty is that this feature engineering requires deep expertise in the field. Recently, deep learning methods such as recurrent neural networks and one-dimensional convolutional neural networks or CNNs have been shown to provide state-of-the-art results on challenging activity recognition tasks with little or no data feature engineering.

# Data Collection
There are many datasets available online to recognize human activities using machine learning techniques. Some of these datasets are as following:
* [Activity Recognition from Single Chest-Mounted Accelerometer Data Set](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer)
* [Activity Recognition system based on Multisensor data fusion (AReM) Data Set](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+system+based+on+Multisensor+data+fusion+%28AReM%29)
* [Daily and Sports Activities Data Set](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities)
* [Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
* [Smartphone-Based Recognition of Human Activities and Postural Transitions Data Set](https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions)
* [Heterogeneity Activity Recognition Data Set](https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition)
* [Intelligent Media Accelerometer and Gyroscope (IM-AccGyro) Dataset Data Set](https://archive.ics.uci.edu/ml/datasets/Intelligent+Media+Accelerometer+and+Gyroscope+%28IM-AccGyro%29+Dataset)
* [Localization Data for Person Activity Data Set](https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity)
* [Smartphone Dataset for Human Activity Recognition (HAR) in Ambient Assisted Living (AAL) Data Set](https://archive.ics.uci.edu/ml/datasets/Smartphone+Dataset+for+Human+Activity+Recognition+%28HAR%29+in+Ambient+Assisted+Living+%28AAL%29)
* [Simulated Falls and Daily Living Activities Data Set Data Set](https://archive.ics.uci.edu/ml/datasets/Simulated+Falls+and+Daily+Living+Activities+Data+Set#)

We will try to reuse some of these datasets for this project. In order to show live demo of this project, we are also planning to collect some data from "Motionsense HRV Wrist Sensor". This sensor has IMU ( 9 DOF - Accelerometer, Gyroscope and Magnetometer ) and a PPG sensor to measure heart-rate.

# Models for Data Processing 
Data from these wearables can be processed using either classical models (Naive Bayes, random forests, SVMs) or neural models (MLP, CNN, LSTM, TCN etc.) to predict 
the activity being performed.

# Human Activities
Some of the human activities which can be targeted under this project are lying, sitting, standing, walking, running, cycling, Nordic walking, watching TV, computer work, car driving, ascending stairs, descending stairs, vacuum cleaning, ironing, folding laundry, house cleaning, playing soccer, rope jumping etc.

# Timeline
* **Week 6:** Data Collection using online sources and wearable.
* **Week 7:** Data Preprocessing and Model Development.
* **Week 8:** Model Validation and Testing for hyperparameter tuning.
* **Week 9:** Activity Log Generation.
* **Week 10:** Report/Video Demo.

# Reference Work
* [Activity recognition using wearable sensors for tracking the elderly](https://link.springer.com/article/10.1007%2Fs11257-020-09268-2).
* [Human Activity Recognition – Using Deep Learning Model](https://www.geeksforgeeks.org/human-activity-recognition-using-deep-learning-model/).
