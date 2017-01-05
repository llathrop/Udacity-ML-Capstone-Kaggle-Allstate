# Machine Learning Engineer Nanodegree
## Capstone Project
Bryan Luke Lathrop 
December 31st, 2016

## I. Definition
_(approx. 1-2 pages)_

### Project Overview
In this section, look to provide a high-level overview of the project in layman's terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_

This Project will be based around the Kaggle competition detailed at: https://www.kaggle.com/c/allstate-claims-severity

This competition centers on the idea that the severity(or cost) of a claim may be predicted based on the several factors in the data set. Much of the work in statistics to date has been used by the insurance industry in pursuit of this goal, and this particular challenge is aimed at recruiting the participants for work in an already tested field.

I choose this competition as it dataset and goals allow us to explore various machine learning techniques without focusing on data collection. I also believe that the techniques and goal used are very close in style to those used in industry currently, and so more applicable to future projects.

### Problem Statement
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

Using the provided dataset, we will use various machine learning techniques to predict the claims severity. Claim severity is expressed as a cost, where a higher cost is a higher severity. The predictions will be made on a per claim basis, and are intended to be applied to future claims as an indicator for customers/agents.

### Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you?ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_

The project success may be evaluated on the improvement in score over the benchmark model, as returned from the competition. Both models will be trained using the same data and submitted for the same test data. As we are using MAE for scoring, we will be looking for the lowest score as the winner. Since we can't verify the test set directly, we will further break out a validation set from the train data, for use as our own test set for the purpose of validating the models before use with the provided test set. This validation set will be sized to about 25% of the train data.

MAE score is defined as the mean of the absolute value of the real minus predicted values of each row in the validation/test data sets: 1/n sum(abs(each_predicted_y-each_real_y))) The advantage of MAE (other than being a contest requirment) is that it provides a simple measurement of the error of a prediction that disregards the sign of the error and doesn't over-emphasize outliers.

Additionally, we will track prediction time for the scores achieved, as well as training time, in an effort to quantify the effort needed to use the score in a production environment. These times will be used with the final scores to determine viability of the model

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

The dataset is provided by the competition organizer, and is anonymized, including removing labels from each data point. We are to assume that this data was gathered in the normal course of the business of prior insurance claims, and will be continue to be gathered so that new predictions may be made. This means that we may not use intuition to provide new features. We are to assume that the data is relevant to the problem and accurate. We may test this relevance, or use methods such as PCA to examine the most relevant labels.

We are provided training and test data set, where the training set includes the "loss" field that we are attempting to predict, and test does not. When looking at the common features, we see 116 categorical and 14 continuous features. The features seem well matched between train and test, with similar mean/standard deviation/min/max. The train set has 188318 rows, and the test set has 125546.

**Data sample:**

    |id  |cat1|cat2|cat3|cat4|cat5|cat6|cat7|cat8|cat9|....|cont6|cont7|cont8|cont9|cont10|cont11|cont12|cont13|cont14|loss
 ---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:----|:----|:----|:-----|:-----|:-----|:-----|:-----|:----|:---
0 |	1|A|B|A|B|A|A|A|A|B|...|0.718367|0.335060|0.30260|0.67135|0.83510|0.569745|0.594646|0.822493|0.714843|2213.18
1	| 2|A|B|A|A|A|A|A|A|B|...|.438917|0.436585|0.60087|0.35127|0.43919|0.338312|0.366307|0.611431|0.304496|1283.60
2	| 5|A|B|A|A|B|A|A|A|B|...|.289648|0.315545|0.27320|0.26076|0.32446|0.381398|0.373424|0.195709|0.774425|3005.09
3	| 10|B|B|A|B|A|A|A|A|B|...|.440945|0.391128|0.31796|0.32128|0.44467|.327915|0.321570|0.605077|0.602642|939.85
4	| 11|A|B|A|B|A|A|A|A|B|...|0.178193|0.247408|0.24564|0.22089|0.21230|0.204687|0.202213|0.246011|0.432606|2763.85


### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

Plan is to present several plots based on data, including a histogram for cont and categorical values, scatterplots for cont, and more.
see DataExploration.ipynb

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

We will use various pre-processing techniques to generate new features and otherwise prepare the data. Following this we will use various sklearn regressors, optimized with grid search, and xgbost to generate the first layer of an ensemble model using a stacking. The following layers will use similar regressions on the output of the prior layers, with the final layer providing a final prediction for each input.

The model will output a predicted 'loss' for each claim in the validation data. Once trained and satisfactory scores are obtained with the validation data, the model will be retrained on the full data set and predictions made on the test data. The result will then be submitted to the Kaggle competition, where a score will be assigned to the model. The solutions score will be evaluated using the mean absolute error(MAE) between the actual and predicted loss.

http://xgboost.readthedocs.io/
http://scikit-learn.org/
https://en.wikipedia.org/wiki/Ensemble_learning#Stacking

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_

The base model for this project is planned as a simple linear regression based on the data, with minimal pre-processing only, and run first with the initial data minus a validation set and once scoring appropriately, submitted for scoring according to the previously mentioned method. This will provide a definitive measurement of the improvement we see in the final model.

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

Several tasks take place 
* the train/validation and test data from the contests is combined 
* categorical data is be transformed to numerical (see LabelEncoder() function), using factorize
* all data is MinMax scaled, 0-1, using the sklean MinMaxScaler() 
* unneeded features will be removed(ex:id)
* data will be transformed to final state for handoff to the model (numpy array)

additionally:
* new features will be created:
    * clusters for each row, for just continous features, for just categorical. clusters are computed by first taking a subset of the      data and calculating the Kmeans cluster for each, with startingClusterSize=int(len(data)*.075). The cluster centers for this are        then used to calculate the number of actual clusters with Meanshift(). Kmeans is then re-run again using this number of clusters.        This is done to provide a significant performance boost over using either one on it's own.
    see :http://jamesxli.blogspot.com/2012/03/on-mean-shift-and-k-means-clustering.html
    * various features may be modified due to relevance.
* features will be analyzed for relevance using PCA(FIXME)

The benchmark model will be trained and tested with and with out these added features

* the data is split back into original train/validation and test segments

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

To start with, we will first pick several normal SKLearn regressor, and combine them into a list for ease of use, along with several parameter/values dicts for use in grid search. We'll then split the provided train dataset into a new train and validation sets, for use in par(80/20 split). Using the new train data, we will run a grid search for each of the selected models, creating a list of regressors with parameters set from the search, and saved to disk for ease of reuse.

We'll follow this by using xgb.cv() to find the best n_rounds for a set of parameters that have been manually selected and experimentally optimized. The train and validation sets are discarded, as we will now work split the entire train set into k-folds(K=5) for the first layer.

for each fold we will use the current fold as the test set and train(or fit) the model on the rest of the data, x, making a prediction, y. The prediction for each fold will be made for each of the regressors that we have selected. The predictions for each fold will then be stacked together for use in the next layer. MAE (Mean Absolute Error) and prediction time are logged so that we may track progress.

Due to run time, first layer models are saved and loaded if present. 

after the predictions are made for the first layer, they are averaged and a cluster is predicted and both added as a feature to the layer.

The same process is followed for layer two, but with input data being the predictions, etc from the first layer. While the average value of the 2nd layer predictions was found to add value, clusters were not at this layer. The predictions from the 2nd layer are then fed to a final regressor for our final train/predict cycle.

At this point, we are able to use the the models trained in each layer to make predictions for the test data, which follows the same cycle to the the third layer, where our final layer is predicted and output.

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model?s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model?s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


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
