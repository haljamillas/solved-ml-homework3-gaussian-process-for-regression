Download Link: https://assignmentchef.com/product/solved-ml-homework3-gaussian-process-for-regression
<br>
<h1>1           Gaussian Process for Regression</h1>

In this exercise, please implement Gaussin process (GP) for regression. The file data gp x.csv and gp t.csv have input data <strong>x </strong>: {<em>x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,…,x</em><sub>100</sub>}<em>,</em>0 <em>&lt; x<sub>i </sub>&lt; </em>1 and the corresponding target data <strong>t </strong>: {<em>t</em><sub>1</sub><em>,t</em><sub>2</sub><em>,…,t</em><sub>100</sub>} respectively. Please take the first 50 points as the training set and the rest as the test set. A regression function <em>y</em>(·) is used to express the target value by

where the noisy signal <em><sub>n </sub></em>is Gaussian distributed,) with <em>β</em><sup>−1 </sup>= 1.

<ol>

 <li>Please implement the GP with exponential-quadratic kernel function given by</li>

</ol>

where the hyperparameters <em>θ </em>= {<em>θ</em><sub>0</sub><em>,θ</em><sub>1</sub><em>,θ</em><sub>2</sub><em>,θ</em><sub>3</sub>} are fixed. Please use the training set with four different combinations:

<ul>

 <li>linear kernel <em>θ </em>= {0<em>,</em>0<em>,</em>0<em>,</em>1}</li>

 <li>squared exponential kernel <em>θ </em>= {1<em>,</em>16<em>,</em>0<em>,</em>0}</li>

 <li>exponential-quadratic kernel <em>θ </em>= {1<em>,</em>16<em>,</em>0<em>,</em>4}</li>

 <li>exponential-quadratic kernel <em>θ </em>= {1<em>,</em>64<em>,</em>32<em>,</em>0}</li>

</ul>

<ol start="2">

 <li>Please plot the prediction result like Figure 6.8 of textbook for training set but one standard deviation instead of two and without the green curve. The title of the figure should be the value of the hyperparameters used in this model. The red line shows the mean <em>m</em>(·) of the GP predictive distribution. The pink region corresponds to plus and minus one standard deviation. Training data points are shown in blue. An example is provided in below.</li>

</ol>

1

<ol start="3">

 <li>Show the corresponding root-mean-square errors</li>

</ol>

for both training and test sets with respect to the four kernels.

<ol start="4">

 <li>Try to tune the hyperparameters by yourself to find the best combination for the dataset. You can tune the hyperparameters by trial and error or use <strong>automatic relevance determination </strong>(ARD) in Chapter 6.4.4 of textbook. (If you implement the ARD method, you will get the bonus points.)</li>

 <li>Explain your findings and make some discussion.</li>

</ol>

<h1>2           Support Vector Machine</h1>

Support vector machines (SVM) is known as a popular method for pattern classification. In this exercise, you will implement SVM for classification. Here, the Tibetan-MNIST dataset is given in x train.csv and t train.csv. Tibetan-MNIST is a dataset of Tibetan images. The input data including three categories: Tibetan 0, Tibetan 1 and Tibetan 2. Each example is a 28×28 grayscale image, associated with a label.

Data Description

<ul>

 <li><strong>x train </strong>is a 300 × 784 matrix, where each row is the first two scaled principal values of a training image.</li>

 <li><strong>t train </strong>is a 300 × 1 matrix, which records the classes of the training images. 0, 1, 2 represent Tibetan 0, Tibetan 1 and Tibetan 2, respectively.</li>

</ul>

In the training procedure of SVM, you need to optimize with respect to the Lagrange multiplier <em>α </em>= {<em>α<sub>n</sub></em>}. Here, we use the Sequential Minimal Optimization to solve the problem. For details, you can refer to the paper [John Platt, “Sequential minimal optimization: A fast algorithm for training support vector machines.” (1998)]. Scikit-learn is a free software machine learning library based on Python. This library provides the sklearn.svm. You are allowed to use the library to calculate the multipliers (coefficients) rather than using the <strong>prediction function </strong>directly.

In this exercise, you will implement SVM based on two kinds of kernel functions

<strong>w </strong>

<ul>

 <li><strong>Linear kernel:</strong></li>

</ul>

<em>k</em>(<strong>x</strong><em><sub>i</sub>,</em><strong>x</strong><em><sub>j</sub></em>) = <em>φ</em>(<strong>x</strong><em><sub>i</sub></em>)<sup>&gt;</sup><em>φ</em>(<strong>x</strong><em><sub>j</sub></em>) = <strong>x</strong><sup>&gt;</sup><em><sub>i </sub></em><strong>x</strong><em><sub>j</sub></em>

<ul>

 <li><strong>Polynomial (homogeneous) kernel of degree 2:</strong></li>

</ul>

<strong>x </strong>= [<em>x</em><sub>1</sub><em>,x</em><sub>2</sub>]

SVM is a binary classifier, but the application here has three classes. To handle this problem, there are two decision approaches, one is the one-versus-the-rest’, and another is the ‘oneversus-one’

<ol>

 <li>Analyze the difference between two decision approaches (one-versus-the-rest and oneversus-one). Decide which one you want to use and explain why you choose this approach.</li>

 <li>Use the dataset to build a SVM with linear kernel to do multi-class classification. Then plot the corresponding decision boundary and support vectors.</li>

 <li>Repeat (2) with polynomial kernel (degree = 2).</li>

 <li>Discuss the difference between (2) and (3).</li>

</ol>

<strong>Hints</strong>

<ul>

 <li>In this exercise, we strongly recommend using matlab to avoid tedious preprocessing occurred in python.</li>

 <li>If you use other languages, you are allowed to use toolbox only for multipliers (coefficients).</li>

 <li>You need to implement the whole algorithms except for multipliers (coefficients).</li>

</ul>

<h1>3           Gaussian Mixture Model</h1>

In this exercise, you will implement a Gaussian mixture model (GMM) and apply it in image segmentation. First, use a <em>K</em>-means algorithm to find <em>K </em>central pixels. Second, use Expectation maximization (EM) algorithm (please refer to textbook p.438-p.439) to optimize the parameters of the model. The input data is hw3 3.jpeg. According to the maximum likelihood, you can decide the color <em>µ<sub>k</sub></em>, <em>k </em>∈ [1<em>,…,K</em>] of each pixel <em>x<sub>n </sub></em>of ouput image 1. Please build a <em>K</em>-means model by minimizing

<em>N          K</em>

<em>J </em>= XX<em>γ</em><em>nk</em>||<em>x</em><em>n </em>− <em>µ</em><em>k</em>||2

<em>n</em>=1 <em>k</em>=1

and show the table of estimated .

<ol start="2">

 <li>Use calculated by the K-means model as means, and calculate the corresponding variances <em>σ<sub>k</sub></em><sup>2 </sup>and mixing coefficient <em>π<sub>k </sub></em>for initialization of GMM</li>

</ol>

Optimize the model by maximizing the log likelihood function log<em>p</em>(<em>x</em>|<em>π,µ,σ</em><sup>2</sup>) through EM algorithm. Plot the log likelihood curve of GMM. (Please terminate EM algorithm when the iteration arrives 100)

<ol start="3">

 <li>Repeat step (1) and (2) for <em>K </em>= 3, 5, 7, and 10. Please show the resulting images in your report. Below are some examples.</li>

 <li>Please show the graph of Log likelihood at different iterations for K = 3, 5, 7, 10 Example are shown below. (This graph is only for your reference.)</li>

 <li>You can make some discussion about what is crucial factor to affect the output image and explain the reason?</li>

 <li>The image shown below is your input image taken by TA, and it is only allowed to be used for homework3.</li>

</ol>