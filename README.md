# Intro to ML

This class was intially designed for the preparatory week of PSL. In this repo, I make some adjustment to fit it to an audience of engineers.
The dedicated slack channel is `#2025-bootcamp`. 

## Lectures

**Day 1**

* 9:00--10:20: (course) Machine learning: recent successes.
  [Machine learning: history, application, successes](./slides/01_machine_learning_successes)

* 10:40-12:00: (course) Introduction to machine learning.
  [Introduction to machine learning](./slides/02_intro_to_machine_learning)

* 14:00--15:30: (course) Machine learning models (linear, trees, neural networks).
  [Supervised machine learning models](./slides/03_machine_learning_models/)

* 16:00-17:30: (course) Scikit-learn: estimation/prediction/transformation.
  [Scikit-learn: estimation and pipelines](./slides/04_scikit_learn/)

## Practical sessions

These practical sessions will necessitate the use of Python 3 with the standard Scipy ecosystem, Scikit-learn and Pytorch. They will make use of Jupyter notebooks. The easiest way to proceed is to have a gmail account and make use of a remote [Google Colab](https://colab.research.google.com/) to run the notebooks.

**Day 1** (at home)
* (lab session) Introduction to Python and Numpy for data sciences.
  - [Python basics](https://colab.research.google.com/github/data-psl/lectures2024/blob/master/notebooks/01_python_basics.ipynb) [(corrected)](https://colab.research.google.com/github/data-psl/lectures2024/blob/master/notebooks/01_python_basics_corrected.ipynb)
  
* (lab session) Practice of Scikit-learn.
  - [Preliminaries](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/01-Preliminaries.ipynb)
  - [intro](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/02.1-Machine-Learning-Intro.ipynb) [(corrected)](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/02.1-Machine-Learning-Intro_corrected.ipynb)
  - [basic principles](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/02.2-Basic-Principles.ipynb)   [(corrected)](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/02.2-Basic-Principles_corrected.ipynb)
  - [SVM](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/03.1-Classification-SVMs.ipynb)  
  - [Regression Forests](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/03.2-Regression-Forests.ipynb)  [(corrected)](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/03.2-Regression-Forests_corrected.ipynb)
  - [PCA](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/04.1-Dimensionality-PCA.ipynb)
  - [Clustering](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/04.2-Clustering-KMeans.ipynb) 
  - [GMM](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/04.3-Density-GMM.ipynb) 
  - [Validation](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/05-Validation.ipynb)  [(corrected)](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/05-Validation_corrected.ipynb)
  - [Pipeline](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/02_sklearn/06-Pipeline.ipynb) 

**Day 2** (at home)
* (lab session) Logistic regression with gradient descent.
  - [Optimization](https://colab.research.google.com/github/data-psl/lectures2024/blob/master/notebooks/03_optimization.ipynb) and the [Corrected notebook](https://colab.research.google.com/github/data-psl/lectures2024/blob/master/notebooks/03_optimization_corrected.ipynb)

**Day 3** (at home)
* (lab session) Classification with Pytorch and GPUs
  - [Notebook 1](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/04_pytorch/01_introduction_to_pytorch.ipynb)
  - [Notebook 2](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/04_pytorch/02_simple_neural_network.ipynb)  [Corrected notebook](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/04_pytorch/02_simple_neural_network_corrected.ipynb)
  - [Notebook 3](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/04_pytorch/03_convolutional_neural_network_mnist.ipynb) [Corrected notebook](https://colab.research.google.com/github/data-psl/lectures2024/blob/main/notebooks/04_pytorch/03_convolutional_neural_network_mnist_corrected.ipynb)


## Additional material (slides)

[Optimization for linear models](https://data-psl.github.io/lectures2024/slides/05_optimization_linear_models/)

[Optimization for machine learning](https://data-psl.github.io/lectures2024/slides/06_optimization_general/)

[Deep learning: convolutional neural networks](https://data-psl.github.io/lectures2024/slides/07_deep_learning/)

[Unsupervised learning](https://data-psl.github.io/lectures2024/slides/08_unsupervised_learning/)

## Acknowledgements

The slides and notebooks were originally written by [Pierre Ablin](https://pierreablin.com/), [Mathieu Blondel](https://mblondel.org/) and [Arthur Mensch](http://www.amensch.fr/).

Some material of this course was borrowed and adapted:
  * The slides from ["Deep learning: convolutional neural networks"](https://data-psl.github.io/lectures2024/slides/07_deep_learning/) are adapted from
  Charles Ollion and Olivier Grisel's [advanced course on deep learning](!https://github.com/m2dsupsdlclass/lectures-labs) (released under the
  [CC-By 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode)).
  * The first notebooks of the scikit-learn tutorial are taken from Jake Van der Plas [tutorial](https://github.com/jakevdp/sklearn_tutorial).

## License
All the code in this repository is made available under the MIT license unless otherwise noted.

The slides are published under the terms of the [CC-By 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode).
