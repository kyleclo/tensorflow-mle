# tensorflow-mle
Just some examples on using TensorFlow to compute MLEs.  

I wanted to explore TensorFlow's versatility as a general-purpose statistical computing library by attempting to fit a range of models with gradient descent.  These include mixture models (traditionally fit using EM algorithm) and random effects models (traditionally fit using explicitly-derived marginal likelihood maximization).  I'll be adding more model implementations from time to time.

See the full blog post at http://kyleclo.github.io/maximum-likelihood-in-tensorflow-pt-1/.

# Installation
This code was written for Python >= 3.6.0.  To install dependencies, run:

```
pip install -r requirements.txt
```

after cloning the repo.
