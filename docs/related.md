
# Related Work



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
* [Final Presentation](???)

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
