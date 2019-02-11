import pandas as pd
import numpy as np

from collections import Counter

class NonParametricClassifier(object):
 
    def __init__(self, alpha = 2, beta = 0, compare = "hellinger", useprior = True):
        
        self.alpha = alpha
        self.compare = compare
        self.useprior = useprior
        self.beta = beta
        self.prior = None
        self.distributions = None
        self.cdfs = None
        self.test_X = None
        self.test_Y = None
        self.predicted = None
        
    def fit(self, df, classes):
        """
        Create empirical CDF from training data. Assumes data is in tidy format.
        CDF is based on power law of 
        
        Keyword arguments:
        df - DataFrame of words and corresponding classes.
        classes - list of class names.
        """
        
        prior = {}
        distributions = {}
        cdfs = {}
        
        all_classes = np.unique(classes)
        
        for cls in classes:
            
            assert cls in df.columns
            
            # create distribution for each class
            
            counts = Counter(df[df[cls] == 1].word)
            norm = sum(counts.values())
            for element in counts:
                counts[element] /= norm
                
            prior.update({cls: len(df[df[cls] == 1].ID.unique())})
            distributions.update({cls: counts})
            cdfs.update({cls: self.get_cdf(counts)})

        norm = sum(prior.values())
        for cls in prior:
            prior[cls] /= norm
            
        self.set_prior(prior)
        self.set_distributions(distributions)
        self.set_cdfs(cdfs)
        
    def set_prior(self, prior):
        self.prior = prior
        
    def set_distributions(self, distribution):
        self.distributions = distribution
        
    def set_cdfs(self, cdfs):
        self.cdfs = cdfs
        
    def set_testX(self, X):
        self.test_X = X
        
    def set_testY(self, Y):
        self.test_Y = Y
        
    def set_predicted(self, predicted):
        self.predicted = predicted
        
    def get_cdf(self, counts):
        tmp = sorted(Counter(counts).items(), key = lambda x: x[1], reverse = True)
        return {tmp[x][0]: x for x in range(len(tmp))}

    def KL_divergence(self, dist1, dist2, alpha):
        
        div = 0

        for element in dist1.keys():
            if element not in dist2:
                div += dist1[element] * np.log(dist1[element] / alpha)
            else:
                div += dist1[element] * np.log(dist1[element] / dist2[element])

        return div

    def hellinger_distance(self, dist1, dist2, alpha, beta):

        num = 0

        for element in dist1.keys():

            if element not in dist2:
                num += ((dist1[element]) ** (1 / alpha) - (beta ** (1 / alpha))) ** alpha
            else:
                num += ((dist1[element]) ** (1 / alpha) - (dist2[element]) ** (1 / alpha)) ** alpha

        num = (1 / np.sqrt(alpha)) * (num ** (1 / alpha))

        return 1 - num
    
    def create_test_data(self, df, classes):
        """
        Create test data CDF's. Assumes data is in tidy format.
        
        Keyword arguments:
        df - DataFrame of words and corresponding classes.
        classes - list of class names.
        """
        test_X = []
        test_Y = []
        
        for cls in classes:
            for song in df.ID.unique():
                counts = Counter(df[(df.ID == song) & (df[cls] == 1)].word)
                if len(counts) == 0:
                    continue
                norm = sum(counts.values())
                for element in counts:
                    counts[element] /= norm
                
                test_X.append(counts)
                test_Y.append(cls)
                
        self.set_testX(test_X)
        self.set_testY(test_Y)
        
    def classify(self, example, print_ = False):
        
        distance = {}
        
        for cls in self.distributions:
            
                if self.compare == "hellinger":
                    tmp = self.hellinger_distance(example, self.distributions[cls], self.alpha, self.beta)
                    distance.update({cls: tmp})
                elif self.compare == "KL":
                    tmp = self.KL_divergence(example, self.distributions[cls], self.alpha)
                    distance.update({cls: tmp}) 
                    
        norm = sum(distance.values())
        for cls in self.distributions:
            distance[cls] /= norm
            
        if self.useprior:
            for cls in self.distributions:
                distance[cls] *= self.prior[cls]
                
        if print_:
            print(distance)
            
        if self.compare == "hellinger":
            return max(distance.items(), key = lambda x : x[1])[0]
        elif self.compare == "KL":
            return min(distance.items(), key = lambda x: x[1])[0]
        
        
    def predict(self, df, classes):
        """
        Classifies new training examples. 
        
        Keyword arguments:
        df - DataFrame of words and corresponding classes.
        classes - list of class names.
        """     
        
        predicted = []
        self.create_test_data(df, classes)
        
        for example in self.test_X:
            predicted.append(self.classify(example))
            
        self.set_predicted(predicted)
            
        return predicted
        