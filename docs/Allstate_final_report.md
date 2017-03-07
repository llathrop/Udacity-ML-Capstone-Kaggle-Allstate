# Machine Learning Engineer Nanodegree
## Capstone Project
Bryan Luke Lathrop - March 7th, 2017

## I. Definition

### Project Overview

This project will be based around the Kaggle competition detailed at: https://www.kaggle.com/c/allstate-claims-severity

This competition centers on the idea that the severity(or cost) of a claim may be predicted based on the several factors in the data set. Much of the work in statistics to date has been used by the insurance industry in pursuit of this goal, and this particular challenge is aimed at recruiting the participants for work in an already tested field.

I choose this competition as it dataset and goals allow us to explore various machine learning techniques without focusing on data collection. I also believe that the techniques and goal used are very close in style to those used in industry currently, and so more applicable to future projects.

### Problem Statement

Using the provided dataset, we will use various machine learning techniques to predict the claims severity. The general type of techniques to be used is know as a "regression". Claim severity is expressed as a cost, where a higher cost is a higher severity. The predictions will be made on a per claim basis, and are intended to be applied to future claims as an indicator for customers/agents. The data does not provide a description of the features, and so we must use information analysis techniques to present a solution. 

### Metrics

The project success may be evaluated on the improvement in score over the benchmark model, as returned from the competition. Both models will be trained using the same data and submitted for the same test data. As we are using MAE for scoring, we will be looking for the lowest score as the winner. Since we can't verify the test set directly, we will further break out a validation set from the train data, for use as our own test set for the purpose of validating the models before use with the provided test set. This validation set will be sized to about 20% of the train data.

MAE score is defined as the mean of the absolute value of the real minus predicted values of each row in the validation/test data sets: 1/n sum(abs(each_predicted_y-each_real_y))) The advantage of MAE (other than being a contest requirment) is that it provides a simple measurement of the error of a prediction that disregards the sign of the error and doesn't over-emphasize outliers.

Additionally, we will review prediction time for the scores achieved, as well as training time, in an effort to quantify the effort needed to use the score in a production environment. These times will be used with the final scores to determine viability of the model

## II. Analysis

### Data Exploration

The dataset is provided by the competition organizer, and is anonymized, including removing labels from each data point. We are to assume that this data was gathered in the normal course of the business of prior insurance claims, and will be continue to be gathered so that new predictions may be made. This means that we may not use intuition to provide new features. We are to assume that the data is relevant to the problem and accurate. We may test this relevance*, or use methods such as PCA to examine the most relevant labels*.

We are provided training and test data set, where the training set includes the "loss" field that we are attempting to predict, and test does not. When looking at the common features, we see 116 categorical and 14 continuous features. The features seem well matched between train and test, with similar mean/standard deviation/min/max. The train set has 188318 rows, and the test set has 125546. Average loss is 3037.3376

**Data sample:**

    |id  |cat1|cat2|cat3|cat4|cat5|cat6|cat7|cat8|cat9|....|cont6|cont7|cont8|cont9|cont10|cont11|cont12|cont13|cont14|loss
 ---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:----|:----|:----|:-----|:-----|:-----|:-----|:-----|:----|:---
0 |	1|A|B|A|B|A|A|A|A|B|...|0.718367|0.335060|0.30260|0.67135|0.83510|0.569745|0.594646|0.822493|0.714843|2213.18
1	| 2|A|B|A|A|A|A|A|A|B|...|.438917|0.436585|0.60087|0.35127|0.43919|0.338312|0.366307|0.611431|0.304496|1283.60
2	| 5|A|B|A|A|B|A|A|A|B|...|.289648|0.315545|0.27320|0.26076|0.32446|0.381398|0.373424|0.195709|0.774425|3005.09
3	| 10|B|B|A|B|A|A|A|A|B|...|.440945|0.391128|0.31796|0.32128|0.44467|.327915|0.321570|0.605077|0.602642|939.85
4	| 11|A|B|A|B|A|A|A|A|B|...|0.178193|0.247408|0.24564|0.22089|0.21230|0.204687|0.202213|0.246011|0.432606|2763.85


### Exploratory Visualization

Plan is to present several plots based on data, including a histogram for cont and categorical values, scatterplots for cont, and more.
see DataExploration.ipynb

here we can see a box plot of the conts values, showing that they are all close in value, but that a few do have outliers.
![box plot - conts](images/DataExplore-BoxPlot-conts.png)

for more about the data,including more stats and visualizations see: [DataExploration.ipynb](https://github.com/llathrop/udacity-ML-capstone-Kaggle-Allstate/blob/master/DataExploration.ipynb)

### Algorithms and Techniques

We will use various pre-processing techniques to generate new features and otherwise prepare the data. Following this we will use various sklearn regressors, optimized with grid search, and xgbost to generate the first layer of an ensemble model using stacking. The following layers will use similar regressions on the output of the prior layers, with the final layer providing a final prediction for each input.

The model will output a predicted 'loss' for each claim in the validation data. Once trained and satisfactory scores are obtained with the validation data, the model will be retrained on the full data set and predictions made on the test data. The result will then be submitted to the Kaggle competition, where a score will be assigned to the model. The solutions score will be evaluated using the mean absolute error(MAE) between the actual and predicted loss.

*  [XGBoost](http://xgboost.readthedocs.io/)
*  [Scikit-learn](http://scikit-learn.org/)
*  [Stacking](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking)

### Benchmark

The benchmark model for this project is planned as a simple linear regression based on the data, with the same pre-processing, and run first with the initial data minus a validation set and once scoring appropriately, submitted for scoring according to the previously mentioned method. This will provide a definitive measurement of the improvement we see in the final model. 

## III. Methodology

### Data Preprocessing

additionally:
* new features will be created:
    * clusters for each row, for just continous features, for just categorical. Clusters are computed by first taking a subset of the      data and calculating the Kmeans cluster for each, with startingClusterSize=int(len(data)*.075). The cluster centers for this are        then used to calculate the number of actual clusters with Meanshift(). Kmeans is then re-run again using this number of clusters.        This is done to provide a significant performance boost over using either one on it's own.The final cluster # is scaled 0-1, to         better fit with regressors that prefer it see: [on-mean-shift-and-k-means-clustering](http://jamesxli.blogspot.com/2012/03/on-mean-shift-and-k-means-clustering.html)
    * various features may be modified due to relevance.
* features will be analyzed for relevance using PCA(FIXME)

The benchmark model will be trained and tested with and with out these added features:
* the data is split back into original train/validation and test segments
* datasets based on subsets of the features: new, conts, cats, orig_only,all_features are generated for each segment needed
* all dataset's are preserved for use.


for details, please review: [preprocess_data.ipynb](https://github.com/llathrop/udacity-ML-capstone-Kaggle-Allstate/blob/master/preprocess_data.ipynb)

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

Taking the data from preprocessing, we use each data set and first pick several SKLearn regressors, and combine them into a list for ease of use, along with several parameter/values dicts for use in grid search. We'll then split the provided train dataset into a new train and validation sets, for use in par(80/20 split). Using the new train data, we will run a grid search for each of the selected models, creating a list of regressors with parameters set from the search, and saved to disk for ease of reuse.

We'll follow this by using xgb.cv() to find the best n_rounds for a set of parameters that have been manually selected and experimentally optimized. The train and validation sets are discarded, as we will now work split the entire train set into k-folds(K=5) for the first layer.

for each fold we will use the current fold as the test set and train(or fit) the model on the rest of the data, x, making a prediction, y. The prediction for each fold will be made for each of the regressors that we have selected. The predictions for each fold will then be stacked together for use in the next layer. MAE (Mean Absolute Error) is logged so that we may track progress.

After the predictions are made for the first layer, they are averaged and a cluster is predicted and both added as a feature to the layer. Due to run time, first layer models are saved and loaded if present.

Predictions are made at this stage for the test and validation set first layer also, and preserved. work for layer 1 is done in: [JustStacking-Layer1.ipynb](https://github.com/llathrop/udacity-ML-capstone-Kaggle-Allstate/blob/master/JustStacking-Layer1.ipynb)

*FIXME* The same process is followed for layer two, but with input data being the predictions, etc from the first layer. While the average value of the 2nd layer predictions was found to add value, clusters were not at this layer. The predictions from the 2nd layer are then fed to a final regressor for our final train/predict cycle.

At this point, we are able to use the the models trained in each layer to make predictions for the test data, which follows the same cycle to the the third layer, where our final layer is predicted and output.
*/FIXME - rewrite for detail above*

### Refinement

Each regression was found to need individual tuning to achieve results, but as we were focused the final output of the stack, more time was spent refining the stacking techniques. It was found that due to the run time of the full model, especially operations like grid search, caching and resuse of the results was perhaps the most important step in the process, with each stage being broken out and and intermediate output of stages being preserved for later use.

At each stage of the process, it was found helpful to provide intermediate results, to ensure that progress has been made.

The base technique of stacking was further refined by adding features via other methods, such as averaging and clustering the results of stage 1 models. Each of these was found to result in a small improvement in final score.

After the initial version of stacking with linear/xgb/ExtraTreesRegressor as the layer 1 models, the largest improvement was found by adding additional models and using them with each variant of the data.


## IV. Results

### Model Evaluation and Validation

The final model is found to provide results inline with any individual regressor for the data. It should be able to withstand input  collected in the same manner, and in fact saw reasonable results when submitted to the competition, falling in the upper third of results, for both public and private Kaggle data sets(Note: the public set is used for testing until the contest concludes, upon which the private set is used to judge scores). It was seen that across all data sets (Validation/Train/Test), the results were similar for similar inputs.

### Justification
In this section, your model?s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_

In the end we have seen that using stacking to combine the results of models has made a significant improvement in scores as compared to the base output of the linear model. This shows that for the type of problem, stacking is a very valid way to arrive at a more accurate solution. 

In our Linear regression benchmark against the full train data we see and an MAE of 1288 for the original data set and 1287 for the data with new features.

In our stacked regression against all train data and the combined datasets we see  an MAE of 1132. 

These scores were also validated via submission to Kaggle. For the linear set, the best score were:(Private)1272.27333, (Public)1264.60772. For the Stacked set, the best score was: (Private)1132.97728, (Public)1119.72524. These match well with our validation set and training set scores.

using these # we can see an approximatly 8-9% improvement in score.

## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you?ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
