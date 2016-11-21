# Machine Learning Engineer Nanodegree

## Capstone Proposal

Bryan Luke Lathrop
Nov 1st, 2016

### Proposal

(approx. 2-3 pages)
<div>

### Domain Background
<i>
(approx. 1-2 paragraphs)

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.</i>

This Project  will be based around the Kaggle competition detailed at:
https://www.kaggle.com/c/allstate-claims-severity

This competion centers on the idea that the severity(or cost) of a claim may be predicted based on the several factors in the data set. Much of the work in statistics to date has been used by the insurance industry in pursuit of this goal, and this particular challenge is aimed at recruiting the particpants for work in an already tested field.

I choose this competition as it dataset and goals allow us to explore various machine learning techniques without focusing on data collection. I also believe that the techiques and goal used are very close in style to those used in industry currently, and so more applicable to future projects.

### Problem Statement
<i>
(approx. 1 paragraph)

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).
</i>

Using the provided dataset we will use various techniques to predict the claims severity. Claim severity is expressed as a cost, where a higher cost is a higher severity. The predictions will be made on a per claim basis.

### Datasets and Inputs
<i>
(approx. 2-3 paragraphs)

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.
</i>

The provided dataset is anonymized, including removing labels from from each data point. We are to assume that this data was gathered in the normal course of the business of prior insurance claims, and will be continue to be gathered so that new predictions may be made.This means that we may not use intuition to provide new features. We are to assume that the data is relevant to the problem and accurate. We may test this relevance , or use methods such as PCA to examine the most relevant labels. 

We are provided training and test data set, where the training set includes the "loss" field that we are attempting to predict, and test does not. When looking the common features, we see 116 categorical and 14 continuous features. The features seem well matched between train and test, with similar mean/standard deviation/min/max. 

### Solution Statement
<i>
(approx. 1 paragraph)

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).
</i>

The model will output a predicted 'loss' for each claim in the test data. This data will then be submitted to the Kaggle competition, where a score will be assigned to the model. The solutions score will be evaluated using the mean absolute error(MAE) between the actual and predicted loss.  

### Benchmark Model
<i>
(approximately 1-2 paragraphs)

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.
</i>

The base model for this project will be a simple linear regression based on the data, with minimal pre-proccessing only, and submitted for scoring according to the previously mentioned method. This will provide a definitive measurement of the improvement we see in the final model.

### Evaluation Metrics
<i>
(approx. 1-2 paragraphs)

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).
</i>

The project success may be evaluated on the improvement in score over the benchmark model, as returned from the competetion. Both models will be trained using the same data and submitted for the same test data. As we are using MAE for scoring, we will be looking for the lowest score as the winner.

Additionaly we will track prediction time for the scores acheived, as well as training time, in an effort to quantify the effort needed to use the score in a production environment.

### Project Design
<i>
(approx. 1 page)

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

</i>

The project may be broken into several categories:
* Data Analysis -  looking for details of the data, such as size, layout, usefulness of features.
* Pre-processing data - transform/scale data as appropriate for the chosen uses.
* Benchmark model - run the benchmark and evaluate the scores
* Final modeling and prediction - use our chosen model and generate a score
* Submission for scoring - submit both Benchmark model and our model for a final score. record score as well as computation times for each model

#### Data Analysis
  We will examine the dataset and ensure that the test and train dataset's are similar enough for use. We will take a look at the features and make predictions about feature correlation, scaling, etc. we will also review outliers in each feature. For categorical data, we will make sure that test data is included in the train data

#### Pre-processing data
 here several tasks will take place:
 * categorical data will be transformed
 * data will be scaled 
 * new features will created (clustering, etc)
 * unneeded features will be removed(ID)
 * data will be transformed to final state for handoff to the model

#### Benchmark model
 The benchmark will be run on a minimally scaled and prepared data set, without new features, etc

#### Final modeling and prediction
 The final model will be trained and validated on the train data. As part of the training, we will use various cross-validation and gridsearch techniques to tune the hyper-params.
 
 The planned final model is expected to be a stacked model using various regressors at each layer. predicted score and train time  will be tracked for each, as well as for the final model

#### Submission for scoring
 The Benchmark and final model will be submitted and the relative score of each recorded.

<i>
Before submitting your proposal, ask yourself. . .

Does the proposal you have written follow a well-organized structure similar to that of the project template?
Is each section (particularly Solution Statement and Project Design) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
Would the intended audience of your project be able to understand your proposal?
Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
Are all the resources used for this project correctly cited and referenced?
</i>

## Submitted on: UNSUBMITTED
