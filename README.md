# Flight-customer-satisfaction-prediction

In this project we have predicted successfully customer satisfaction level for flight customer based on certain features like customer_type, Flight distance, age, gender, seat comfort, cleanliness etc.

# Introduction

Our dataset is retrieved from www.kaggle.com/teejmahal20/airline-passenger-satisfaction,
which was originally collected by T J Klein and uploaded on Kaggle.com. It is open for any public usage/research 
while the source of the data is anonymous. This dataset was collected as a part of machine learning project 
to determines what features best predict customer satisfaction. It was last updated on February 20, 2020. 
The dataset contains the airline passenger survey data to predict features correlated with passenger satisfaction, 
which is saved in “.CSV” file format. There are two files: training data and test data. Altogether there are 25 columns and 129,880 rows.
Many of the columns are ordinal factors that ask passengers to rank their level of satisfaction with a particular aspect of their trip,
such as inflight service or convenience. 


# MODELING TECHNIQUES
After preparing the data, we can implement several tree-based methods and set performance metrics to evaluate their performance with our passenger data.
A.   Models    
1)	Logistic Regression
Logistic regression is a classification algorithm used to find the probability of success or failure of certain event. It describes the relationship between one dependent variable and one or more independent variables. Logistic regression is easy to implement, interpret, and efficient to train. It performs well when the data set is linearly separable and can easily extend to multiple classes and a probabilistic view of class predictions. 
2)	Decision Tree
Decision Tree is a type of supervised machine learning where the data is continuously split according to a certain parameter. It shows the process of decision and easy to interpret and understand the outcome with an abridged description. Decision Tree works well with categorical variables without encoding. It is also useful handling missing values in training set or testing set as it categorizes missing values as one category. 
3)	  Random Forest
A random forest model is an ensemble machine learning algorithm that creates multiple uncorrelated decision trees by taking a bagging approach to sampling along. It differs from bagging as it takes a random selection of features as opposed to all of them. The collective results from these trees are then used to make a classification for a given datapoint. Random forests is very accurate dealing with non-linear data as well as outliers and works well with large datasets. Conversely though, the training process is notably slow. 
4)	 Bagging
Bagging, which stands for “bootstrap aggregation”, is an ensemble method that creates a number of learner models. These models run in parallel with each other and the overall average of their results is averaged with equal weighting in order to make a prediction. By taking the average result, this approach is able to reduce the variance that comes from a simple decision tree model. However, it does not reduce bias. The equal weighting method too may also result in trees that have no real impact being used for the prediction. With this in mind, we explore how a bagging approach can help us determine whether a given passenger is satisfied or not satisfied.
5)	Gradient Boosting Machine
A GBM approach uses boosting and applies a gradient descent algorithm to minimize the loss function. It has been noted to have a very high predictive power, even more so than a random forest. GBM has a lot of flexibility since it enables optimization on different loss functions with various hyperparameters tuning options. On the flip side, despite its strong predictive performance, it could result in overfitting. Additionally, although flexible, it is very complex, computationally expensive, and less interpretable than other models.
6)	Neural Network
A Neural Network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In this sense, neural networks refer to systems of neurons, either organic or artificial in nature. Neural Networks can adapt to changing input, so it generates the best possible result without needing to redesign the output criteria.
Artificial Neural Networks (ANN) are comprised of simple elements, called neurons, each of which can make simple mathematical decisions. Together, the neurons can analyze complex problems, emulate almost any function including very complex ones, and provide accurate answers. A shallow neural network has three layers of neurons: an input layer, a hidden layer, and an output layer. 



# EXPERIMENT SETTINGS
1)	Logistic Regression
A unique function called run_model is defined. The aim of this function is to run the logistic regression and populate results, such as the ROC curve, to help analyze the model. Finally, the logistic regression is executed. It is important to note that regularization is applied to the logistic regression by default. In this case though, we specifically specify for the model to use an elastic net regularization.
 
 
	From the summary of the logistic regression we see that the accuracy rate of the model was about 0.853, and the ROC are under the curve was about 0.849.
 
2)	Decision Tree
    Using the DecisionTreeClassifier package, we create a decision tree with a max depth of 12. We take the square root of the total number of features to determine the number of features used for the split.
 
 
We see that the results from the decision tree are more accurate than the logistic regression with an accuracy rate of about 0.923.
3)	Random Forest
A function is defined to easily call and evaluate our model performance by displaying accuracy information and plots. An example of this is shown below:
 
       The number of trees to use is defined in the variable n_estimators_hyp, and the minimum number of samples required to split an internal node is defined in the variable min_sample_leaf_hyp.
 
       The RandomForestClassifier function from the sklearn library is called within a nested for-loop to create the models with each of the parameters. For example, a random forest with 100 trees is created with a minimum of 10 leaves. A few iterations of the models are collected and shown below with their Out-of-Bag score.
 
 
       The OOB score represents the accuracy rate for based on the number of correctly predicted values from the out-of-bag sample. With this information, a random forest model is created using 400 trees and a minimum of sample leaf of 1 is stored. The OOB score for this model is approximately 0.9632.
 
The testing set is then used with this model to see the results of the prediction.
 
 
 Calling our custom eval_result function allows us to see the overall accuracy of this random forest model. The training accuracy for the testing dataset is approximately 96.35% and the area under the ROC curve for it is about 96.13%.
 
 
4)	Bagging
        The BaggingClassifier from the sklearn library is used to create our bagged decision trees, and the RepeatedStratifiedKFold and GridSearchCV packages are used to focus on cross validiation and to fit the best model to the training set based on a set of parameters.
 
       Here, the model is created after calling BaggingClassifier. The random state is set so that the results can be reproducible. The number of trees is represented by n_estimators, and cv represents a ten-fold cross validation is repeated three times. These are used as parameters for the GridSearchCV function. Afterwards, the model is fitted with the training dataset, and we find that the best score from this model is about 0.962.






5)	Gradient Boosting Machine
      The lightgbm function is used to create a gradient boosting machine.
 
 
     With a gradient boosting machine, we had an accuracy rate of about 0.962. 

6)	Neural Network
We used TensorFlow Library to demonstrate the Neural Network model. Libraries from the Keras package is called in order to create the neural network.  
 
 
       Since neural networks require a series of layers, the Sequential function is called to initialize the model. Afterwards, the input layer is created to take in our 21 inputs and then the first hidden layer is created with 128 nodes. A total of 5 hidden layers are created with 64, 32, 16, and 8 layers respectively. The Rectified Linear Unit activation function is used for all hidden layers as it has been noted to achieve generally better performance than the sigmoid and tanh functions. 
 
       The model is then compiled with a cross entropy loss function for binary classification problems. The Adam algorithm is used as the optimizer. This is a gradient descent method that automatically tunes itself and adapts the learning rates as the model develops. 
 
       Here, the model is fitted using the training data. An epoch of 5 and batch size of 34 is arbitrarily chosen. In other words, the network will go through the training dataset 5 times and take a sample of 𝑁/34 rows in order to develop the model and update its parameters, where N is the total number of rows in the dataset. We have 103,904 rows altogether in the training dataset, so a batch size of 34 is chosen since it evenly divides it. The results are shown below: 
 
       In this execution, running 5 epochs resulted in the highest accuracy rate. The accuracy rate is very close to 1, indicating that this neural network model is performing very well. As for time, it took about 48 seconds to run each epoch, so it was relatively quick.
       The model is then performed with the test data and the values are cleaned to ensure that they represent binary responses. 
 
Calling the accuracy_score function shows that we have an accuracy rate of about 96%.  
Models Optimization

6.1)	Neural Network – Hyper Parameter Tuning
The model has already been fairly optimized since the ReLU activation function allows for a generally high performance and Adam algorithm allows for automatic tuning. To further explore model optimization using Hyper Parameter Tunings, several models are created to explore how altering different components can affect the results. First, we experiment with the epoch and batch size values. An epoch of 4 for instance produced the following result: 
  
       The overall accuracy and loss values with this implementation seem to be more consistent than the original model, but the overall results are very similar. Ideally, we would be able to reduce the number of epochs and maintain accuracy while reducing the time it takes to run.
       Next, we explore batch size. A lower batch size would mean that we would need more samples for each epoch iteration. This would increase the time it would take to run. Because the model is already very accurate, the batch size is increased to see if we can maintain accuracy while also decreasing the time. 
 
       Using a batch size of 68 significantly reduced the time it took to fit the model with 5 epochs. Notably, we also see a very slight increase in accuracy. Since there is an improvement in accuracy and time, this batch size is kept.
       Then we explore the learning rate by using the SGD library. The default learning rate in the Keras package is 0.01. We decrease it to 0.001 to see if a lower rate will improve the model. 
 
       The time improved by a few seconds, but the accuracy suffered with this new learning rate. It is important to note though that smaller learning rates may need more training epochs to prevent the model from converging too quickly to a solution. If we choose not to use an adaptive learning rate, then we will need to alter the other parameters to see what works best with a specific learning rate.
       Finally, we see how the number of layers and nodes can affect the model. We take out a layer and update the number of nodes. 
         All other factors are then left as the original model and then compiled. The running time is less but the accuracy also decreases from our original model.  
         
         


# CONCLUSION AND FUTURE SCOPE

The logistic regression was especially insightful in making predictive classifications since it was able to optimize the model through an elastic net regularization. Going forward, we can use these models to better understand the initial problem of the dataset and how to best determine whether a passenger is satisfied with their travel experience or not.
From the comparison we could see that when increasing the batch size of Neural Network model, our prediction gets the highest accuracy of 97%. The Logistic Regression takes the a short amount of time in the model training process but the accuracy is the lowest. The decision tree is extremely fast and is fairly accurate, which suggest that a more simplistic model may be the most efficient in these cases. Meanwhile, the Random Forest model also gets a high accuracy while it takes much less time than Neural Network and Bagging model. Since it is important to consider the tradeoff between accuracy and the running time, in this project we would recommend predicting the customer airline satisfaction using the Random Forest method. 

 

