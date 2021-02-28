# Federated Principal Component Analysis (FPCA)
 
# I. Abstract

Federated Principal Component Analysis (FPCA) is proposed to compute the principal components of
a given data set in a distributed fashion where there are several workers (distributed devices) and one
master (data center). This framework is called federated learning, which allows a user to attain the global
model without sharing the data among the local workers. This idea is essential in protecting data privacy.

<p align="center">
  <img width="50%" height="50%" src="/img/master-worker_scenario.jpg">
</p>


# II. Method

An optimization algorithm called augmented direction method of multipliers is used (ADMM) to achieve federated learning. To start with, The objective function of PCA can be formulated as:

![PCA_Formula](/img/PCA_formula.png)

Then by ADMM we can derive the worker’s and master’s algorithms, where z is denoted as the global model and u is the dual variable from ADMM:

<p align="center">
    <img src="/img/worker_algorithm.png" width="550"/>
    <img src="/img/master_algorithm.png" width="550"/>
</p>

We can illustrate the idea of FPCA by the toy example below:

<p align="center">
    <img src="/img/toy_iter0.png" width="450"/>
    <img src="/img/toy_iter10.png" width="450"/>
</p>

In the toy example, there are 10 workers each having their own share of data. The dots scattered on the plot is the global data, assuming not seen by all the workers. The thick yellow line represents the first global PC and the other 10 lines stand for the first PCs calculated by each worker. At the begining (iter = 0), it can be observed that the PC each worker initialzes is stretching toward differnt direction from each other. However, at iteration 10 the workers already seem to reach a consensus solution without seeing the whole data throughout the whole process. 

# III. Experiments

Then we compare our FPCA method to the SVD method, which is currently the most common way to find PCs. The evaluation metric is the cosine similarity between the first PC we derived by FPCA and the one by SVD. Each experiment is repeated 10 times and presented in the convergence plots below:

<p align="center">
    <img src="/img/exp_pc1_10.png" width="80%"/>
<p>

# IV. Conclusion
In our work PCA is adapted into a federated learning setting, which not only solves the distributed machine learning problem but also ensure data privacy. The federated PCA is built up by formulating the projection approximation method as a consensus optimization problem and solving it by ADMM. In the application scenario, the data is distributed across multiple workers who compute and submit their model to a master functioning as the center integrating and broadcasting back the models to the workers. During the training process, the data is always kept private and unshared, which guarantees the data privacy. In addition to that, when processing a large dataset that can't be stored in a single device, the federated setting also allows us to conduct PCA by computing on multiple devices. Through a series of experiment it's found that the first PC calculated by FPCA is consistent with the one calculated by SVD method, which proves our method valid.
