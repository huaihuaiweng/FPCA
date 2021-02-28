# FPCA
 
I. Abstract

Federated Principal Component Analysis (FPCA) is proposed to compute the principal components of
a given data set in a distributed fashion where there are several workers (distributed devices) and one
master (data center). This framework is called federated learning, which allows a user to attain the global
model without sharing the data among the local workers. This idea is essential in protecting data privacy.

![worker-master_scenario](/img/master-worker_scenario.jpg)

II. Method

An optimization algorithm called augmented direction method of multipliers is used (ADMM) to achieve federated learning. To start with, The objective function of PCA can be formulated as:

![PCA_Formula](/img/PCA_formula.png)

Then by ADMM we can derive the worker’s and master’s algorithms, where z is denoted as the global model and u is the dual variable from ADMM:

<img align="left" width="100" height="100" src=/img/worker_algorithm.png>

<img align="left" width="100" height="100" src=/img/master_algorithm.png>

III. EXperiments

Then we compare our FPCA method to the SVD method, which is currently the most common way to find PCs. The evaluation metric is the cosine similarity between the first PC we derived by FPCA and the one by SVD. Each experiment is repeated 10 times and presented in the convergence plots below:

