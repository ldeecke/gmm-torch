import numpy as np
import torch
from gmm import GaussianMixture

import unittest


class CpuCheck(unittest.TestCase):
    """
    Basic tests for CPU models.
    """

    def testPredictClasses(self):
        """
        Assert that torch.FloatTensor is handled correctly.
        """

        x = torch.randn(4, 2)
        n_components = np.random.randint(1, 100)

        model = GaussianMixture(n_components, x.size(1))
        model.fit(x)
        y = model.predict(x)

        # check that dimensionality of class memberships is (n)
        self.assertEqual(torch.Tensor(x.size(0)).size(), y.size())


    def testPredictProbabilities(self):
        """
        Assert that torch.FloatTensor is handled correctly when returning class probabilities.
        """

        x = torch.randn(4, 2)
        n_components = np.random.randint(1, 100)

        model = GaussianMixture(n_components, x.size(1))
        model.fit(x)

        # check that y_p has dimensions (n, k, 1)
        y_p = model.predict(x, probs=True)
        self.assertEqual(torch.Tensor(x.size(0), n_components, 1).size(), y_p.size())


class GpuCheck(unittest.TestCase):
    """
    Basic tests for GPU models.
    """

    def testPredictClasses(self):
        """
        Assert that torch.cuda.FloatTensor is handled correctly.
        """

        x = torch.randn(4, 2).cuda()
        n_components = np.random.randint(1, 100)

        model = GaussianMixture(n_components, x.size(1)).cuda()
        model.fit(x)
        y = model.predict(x)

        # check that dimensionality of class memberships is (n)
        self.assertEqual(torch.Tensor(x.size(0)).size(), y.size())


    def testPredictProbabilities(self):
        """
        Assert that torch.cuda.FloatTensor is handled correctly when returning class probabilities.
        """

        x = torch.randn(4, 2)
        n_components = np.random.randint(1, 100)

        model = GaussianMixture(n_components, x.size(1))
        model.fit(x)

        # check that y_p has dimensions (n, k, 1)
        y_p = model.predict(x, probs=True)
        self.assertEqual(torch.Tensor(x.size(0), n_components, 1).size(), y_p.size())


if __name__ == "__main__":
    unittest.main()
