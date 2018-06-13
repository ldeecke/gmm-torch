This repository contains an implementation of a simple **Gaussian mixture model** (GMM) fitted with Expectation-Maximization in [pytorch](http://www.pytorch.org). The interface is closely with that of [sklearn](http://scikit-learn.org).

A new model is instantiated by calling `m = gmm.GaussianMixture(n_components, d)`. Once instantiated, the model expects tensors in a flattened shape `(n, d)`. Predicting class memberships is straightforward, first fit the model via `m.fit(data)`, then predict with `m.predict(data)`.

Some sanity checks may be executed by calling `python test.py`. To handle data on GPUs, do not forget to call `m.cuda()`.