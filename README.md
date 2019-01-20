# modelfitter
Fit your sklearn models on cv.

The CVModelFit class takes input
* X(features) and y(target)
* A scikit-learn model
* And a scikit-learn cv fold generator.

And fits those models on each cv fold of the data, hopefully outputting a more generalized prediction.

The score() method should be called with a scikit-learn metric.
