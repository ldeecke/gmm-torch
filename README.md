This repository contains an implementation of a simple **Gaussian mixture model** (GMM) fitted with the Expectation-Maximization algorithm in [pytorch](http://www.pytorch.org). The interface is closely aligned to that of [sklearn](http://scikit-learn.org).

A new model is instantiated by calling `gmm.GaussianMixture(n_components, d)`. Once instantiated, the model expects tensors in a flattened shape `(n, d)`. Predicting class memberships is straightforward, first fit the model via `gmm.GaussianMixture.fit(data)`, then predict with `gmm.GaussianMixture.predict(data)`.

Some sanity checks may be executed by calling `python test.py`. To handle data on GPUs, do not forget to call `gmm.GaussianMixture.cuda()` on the instantiated model.