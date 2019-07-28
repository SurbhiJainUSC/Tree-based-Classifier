# Tree-based-Classifier
The task is to classify failure of truck components using tree-based methods like Random Forest Classifier and
Logistic Model Trees.

## Dataset
The system in focus is the Air Pressure system (APS) which generates pressurised air that are utilized in various 
functions in a truck, such as braking and gear changes. 
The datasets' positive class consists of component failures for a specific component of the APS system. 
The negative class consists of trucks with failures for components not related to the APS.
The APS Failure at Scania Trucks dataset contains a training set and a testing set. 
The training set contains 60000 examples in total in which 59000 belong to the negative class and 1000 positive class. 
The test set contains 16000 examples. The instances are descibed by 171 attributes (numeric) including the class label.

## Data Preprocessing
The data set has missing values. Various techniques that are used to handle missing values:
<li> Substitute missing values with a constant (distinct from all other values) that has a meaning in that domain. <br/>
<li> Substitute missing values with the value of the randomly selected observation. <br/>
<li> Use statistics such as Mean, Median, Mode to fill out missing values. <br/>
<li> Use predictive models where missing values are treated as output of the predictive model and are predicted based 
on other data points that do not have missing values. <br>
<li> Use iterative methods based on Expectation Maximization to handle missing values. <br/>

All the missing values in the dataset are replaced by attribute mean.
Correlation matrix is used to find out correlation/association between two variables in the dataset. 
Also, Coefficient of Variation (CV) is used to measure relative variability of the variables.

## Random Forest Classifier
Random Forest Classifier is an ensemble method that creates a set of decision trees from randomly selected subset of 
training set. It then aggregates the votes from different decision trees to decide the final class of the test data point.
Random forest classifier model has been trained on the dataset with out-of-bag estimation to obtain generalized accuracy.

## Synthetic Minority Oversampling Technique (SMOTE)
The dataset is highly imbalanced because there are 74625 observations that belong to negative class, 
and only 1375 observations belong to class positive. If we apply any traditional classifier on imbalanced dataset, 
they are likely to predict everything as negative (the majority class). So, SMOTE has been used in which the minority 
class is over-sampled by creating synthetic examples rather than by over-sampling with replacement.

## Logistic Model Trees (LMT)
LMT is a classification model with an associated supervised training algorithm that combines logistic regression 
and decision tree learning methods. Tree growing starts by building a logistic model at the root using the 
LogitBoost algorithm. The number of iterations is determined using 5-fold cross-validation. Scikit-learn is used to call
Weka to train LMT for classification. LMT has been trained on both class imbalanced data and data balanced after applying
SMOTE. 
