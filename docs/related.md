
# Related Work
Human activity recognition has gained importance in recent years due to its applications in various fields such as health, security and surveillance, entertainment, and intelligent environments. A significant amount of work has been done on human activity recognition and researchers have leveraged different approaches, such as wearable, object-tagged, and device-free, to recognize human activities. 

Classical approaches to the problem involve hand crafting features from the time series data based on fixed-size windows and training machine learning models, such as ensembles of decision trees. The difficulty is that this feature engineering requires deep expertise in the field. Recently, deep learning methods such as recurrent neural networks and one-dimensional convolutional neural networks or CNNs have been shown to provide state-of-the-art results on challenging activity recognition tasks with little or no data feature engineering.

In  "[Human Activity Recognition: A Survey](https://www.sciencedirect.com/science/article/pii/S1877050919310166)" work,  the authors  present  various  state-of-the-art  methods  used for HAR and  describe  each  of  them  by  literature  survey.  They use different datasets for each of the methods wherein the data are collected by different means such as sensors, images, accelerometer, gyroscopes, etc. and the placement of these devices at various locations. This survey concludes that there is no single method which is best for recognition of any activity, hence in order to select a particular method for the desired application, one needs to take various factors into consideration and determine the approach accordingly.

In order to come up with lifestyle advice towards the elderly, [Activity recognition using wearable sensors for tracking the elderly](https://link.springer.com/article/10.1007%2Fs11257-020-09268-2) uses HAR to quantify their lifestyle, before and after an intervention. This research focuses on the task of activity recognition (AR) from accelerometer data. They collect a substantial labelled dataset of older individuals wearing multiple devices simultaneously and performing a strict protocol of 16 activities. Using this dataset, they train Random Forest AR models, under varying sensor set-ups and levels of activity description granularity. Their model combines ankle and wrist accelerometers and produces results with an accuracy of more than 80% for 16-class classification.

Most of these works are using some sort of IMU data from the sensors located at multiple locations. Some of these sensors are not available to users in real-time settings which makes such works impractical. Therefore, in our project, we are targeting HAR using only a wristwatch worn on the dominant hand. Compared to other works, this is practical as smart watches are very common nowadays. This decision to use only one sensor introduced a lot of difficulties in the project. We used various techniques to fix these difficulties, as discussed in the [Technical Approach](https://gargbruin.github.io/WALG/technical.html) section.

# Strengths of this work

* As shown in Related Work section, most of the work in this area uses a combinations of sensors at multiple locations (ankle, chest, wrist etc). Most of these sensors are not available to users in real-time. Therefore, we are targeting human activity detection using only a wrist watch which is very common nowadays (According to Pew Research Center, [About one-in-five Americans use a smart watch or fitness tracker](https://www.pewresearch.org/fact-tank/2020/01/09/about-one-in-five-americans-use-a-smart-watch-or-fitness-tracker/)). Combination of multiple sensors provides a good set of distinguishing features which makes it easy for the model to detect multiple activities. Whereas, in this project, the limited set of sensors made it difficult for our model to distinguish between activities. To fix this problem, we have used multiple approaches such as merging similar activities, adding untargeted activities, adding features like mean and standard deviation etc. 
* Our final trained model is very small in size and can therefore be easily implemented on the wrist watch itself. Such prediction on edge devices is also beneficial from a privacy point of view.

# Weakness of this work

* Since, we are targeting detection with IMU data collected from only a wristwatch, we don't have enough distinguishing features to target a large number of activities. In order to increase the number of targeted activities, we will need to add data from more sensors like magnetometer, heart-rate monitor.
* Since we are using a machine learning approach, our model's predicition accuracy on real-time data heavily depends on the training data. If the user in real-time setting performs the activity a bit differently as compared to the training data, the model might fail to predict the activity correctly. This problem can be solved by adding more and more data from different users but that will also require a bigger model which might be difficult to implement on edge devices. 

# Future Directions

* We are targeting a limited number of activities in our project. The number of targeted activities can be increased by adding data from more sensors like magnetometer, heart-rate monitor. We couldn't try this because we were not able to get training datasets with data from these sensors or it was not easy to extract some of these sensors' data (heart-rate monitor) from Apple Watch.
* New advancements in dataset preprocessing or machine learning approaches can be applied to improve the model's accuracy.

# Individual Contribution
We both worked together for the complete project either through in-person meetings or Zoom calls.

# Presentation Links

* [Midterm Presentation](https://github.com/gargbruin/WALG/blob/main/Presentations/Midterm%20Presentation.pptx)
* [Final Presentation](https://github.com/gargbruin/WALG/blob/main/Presentations/Final%20Presentation.pptx)

# Acknowledgement
We would like to thank J.Vikranth Jeyakumar for his lecture and tutorial on "Human Activity Recognition using Deep Learning (Tensorflow)" and sharing his source code which we referred for our project.

# References
Human Activity Recognition (HAR) is an active area of research because of it's importance in multiple applications. Some of the recent works in this area are as follows: 

* [Human Activity Recognition: A Survey](https://www.sciencedirect.com/science/article/pii/S1877050919310166).  
* [Human activity recognition: A review](https://ieeexplore.ieee.org/document/7072750).  
* [Activity recognition using wearable sensors for tracking the elderly](https://link.springer.com/article/10.1007%2Fs11257-020-09268-2).  
* [Human Activity Recognition – Using Deep Learning Model](https://www.geeksforgeeks.org/human-activity-recognition-using-deep-learning-model/).  
* [Human Activity Recognition using CNN](http://www.ijsrp.org/research-paper-0220.php?rp=P989628).  
* [Deep learning algorithms for human activity recognition using mobile and wearable sensor networks: State of the art and research challenges](https://www.sciencedirect.com/science/article/pii/S0957417418302136).  
* [Evaluate Machine Learning Algorithms for Human Activity Recognition](https://machinelearningmastery.com/evaluate-machine-learning-algorithms-for-human-activity-recognition/)
