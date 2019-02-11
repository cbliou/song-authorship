import pandas as pd
import numpy as np

from collections import Counter

class CDFClassifier(object):
 
    def __init__(self, smooth = 2, compare = "diff"):
        
        self.smooth = smooth
        self.compare = compare
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
                
            distributions.update({cls: counts})
            cdfs.update({cls: self.get_cdf(counts)})

        self.set_distributions(distributions)
        self.set_cdfs(cdfs)
        
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
    
    def get_ranks(self, cdf, dist):
        """
        Get ranks based on empirical CDF.
        """
        dist = sorted(dist.items(), key = lambda x: x[1], reverse = True)
        return [cdf[x[0]] if x[0] in cdf else len(cdf) / self.smooth for x in dist]
    
    def create_training_data(self, df, classes):
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
        
    def classify(self, example):
        distance = {}
        
        for cls in self.cdfs:
            compare = self.get_ranks(self.cdfs[cls], example)
            if self.compare == "diff":
                distance.update({cls: sum(np.diff(compare) > 0)})
            elif self.compare == "Mann-Whitney":
                distance.update({cls: sum(compare)})
            
        if self.compare == "diff":
            return max(distance.items(), key = lambda x : x[1])[0]
        elif self.compare == "Mann-Whitney":
            return min(distance.items(), key = lambda x: x[1])[0]
        
        
    def predict(self, df, classes):
        """
        Classifies new training examples. 
        
        Keyword arguments:
        df - DataFrame of words and corresponding classes.
        classes - list of class names.
        """     
        
        predicted = []
        self.create_training_data(df, classes)
        
        for example in self.test_X:
            predicted.append(self.classify(example))
            
        self.set_predicted(predicted)
            
        return predicted
        