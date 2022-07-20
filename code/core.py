import torch

class SmoothClassifier():
    def __init__(self, base_classifier, num_classes, sigma):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
    
    def certify(self):
        self.base_classifier.eval()

        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError
    
    def sample_noise(self):
        raise NotImplementedError
