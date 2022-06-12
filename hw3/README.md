## HW3Q1
* This contains a multilayer perceptron (MLP) network.
* Data:
    * 4 classes were used with uniform priors.
    * Gaussian class-conditional pdfs for a 3-dimensional real-valued random vector X was specified.
    * Training and testing datasets were generated from this gaussian mixture model (GMM).
* MLP Structure:
    * A 2-layer MLP (1 hidden layer of perceptrons) was used.
    * The optimal number of perceptrons were selected using cross-validation.
    * ReLU activation function was used in 1st layer. 
    * Softmax function was used in output layer.
* Theoretical Optimal Classifier:
    * A theoretically optimal classifier was applied on the test dataset using minimum-probability of error classification. This provided the aspirational performance level for the MLP classifier.