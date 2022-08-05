import math
import numpy as np
import torch
from scipy.stats import binom_test, norm
from statsmodels.stats.proportion import proportion_confint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SmoothClassifier(object):
    ABSTAIN = -1

    def __init__(self, base_classifier, num_classes, sigma):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
    
    def certify(self, x: torch.tensor, N0: int, N: int, alpha: float, batch_size: int) -> (int, float):
        """
        Monte Carlo algorithm for certifying that g's prediction is robust around x within some L2 radius. With probability at least 1-alpha, the class returned by this method (say g_hat(x))will equal the actual g(x) and g's prediction will be robust within an L2-ball of radius R around x

        x: input tensor [channel x height x width]
        N0: number of Monte Carlo noise samples for class prediction/selection
        N: number of Monte Carlo noise samples for robust radius estimation
        alpha: maximum bound on the probability of wrong prediction (i.e, P(g(x) != g_hat(x)) <= alpha)
        """
        self.base_classifier.eval()

        class_counts = self.get_noise_count_per_class(x, N0, batch_size)
        pred_class_idx = np.argmax(class_counts)
        # print("class predicted: ", pred_class_idx)

        class_counts = self.get_noise_count_per_class(x, N, batch_size)
        nA = class_counts[pred_class_idx]
        # print("second time done: ", nA)

        pA_lower_bar = proportion_confint(nA, N, alpha=2 * alpha, method="beta")[0]

        if pA_lower_bar < 0.5:
            return SmoothClassifier.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pA_lower_bar)
            return pred_class_idx, radius


    def predict(self, x: torch.tensor, N: int, alpha: float, batch_size: int) -> int:
        """
        Monte Carlo algorithm for evaluating the prediction of g at x = g_hat(x). With probability at least 1-alpha, the class returned by this function would equal g(x)

        x: input [channel x height x width]
        N: number of noise samples to use for prediction
        alpha: maximum bound on the probability of wrong prediction (i.e, P(g(x) != g_hat(x)) <= alpha)

        returns: the predicted class label or "abstain"
        """

        self.base_classifier.eval()
        
        class_count = self.get_noise_count_per_class(x, N, batch_size)
        top2_class_idx = np.argsort(class_count)[::-1][0:2]

        class_count_first_highest = class_count[top2_class_idx[0]]
        class_count_second_highest = class_count[top2_class_idx[1]]

        if binom_test(class_count_first_highest, class_count_first_highest + class_count_second_highest, p=0.5) > alpha:
            return SmoothClassifier.ABSTAIN
        else:
            return top2_class_idx[0]

    def get_noise_count_per_class(self, x: torch.tensor, num_noise_samples: int, batch_size: int) -> np.ndarray:
        """
        sample base classifier's prediction under noisy corruptions of input x

        x: input sample [channels x height x width]
        num_noise_samples: number of noise samples to use for class selection

        returns: an ndarray of length num_classes containing the per-class count
        """
        with torch.no_grad():
            noise_count_per_class = np.zeros(self.num_classes, dtype=int)

            samples_to_generate = num_noise_samples
            for _ in range(math.ceil(samples_to_generate / batch_size)):

                batch_size_itr = min(samples_to_generate, batch_size)
                samples_to_generate -= batch_size_itr

                x_repeated = x.repeat((batch_size_itr, 1, 1, 1))
                noise_samples = torch.randn_like(x_repeated, device=device) * self.sigma
                corrupted_samples = x_repeated + noise_samples
                # print(corrupted_sample.shape)
                # corrupted_sample = corrupted_sample.to(device)
                # print(x.get_device())

                predictions = torch.argmax(self.base_classifier(corrupted_samples), dim=1)
                # print(predictions.shape, predictions.max(), predictions.min())

                for class_idx in predictions:
                    noise_count_per_class[class_idx] += 1

        assert(num_noise_samples == np.sum(noise_count_per_class))

        return noise_count_per_class