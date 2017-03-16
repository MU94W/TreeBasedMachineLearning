"""
Author: MU94W
Date:   2017-03-16
"""

def calDist(vec1, vec2):
    return sum([diff ** 2 for diff in (vec1 - vec2)]) ** 0.5

class USL(object):
    def __init__(self):
        """
        Object properties:
        dots: the number of dots (to be) clustered.
        dims: the number of features of a single dot.
        cate_num: the number of centroids/medoids.
        data: the features of all the dots.(numeric)
        name: a list that contains the dots' names. It may be a path or the coordinate of a dot.
        cateDotSets: a list, it contains '%d' % cluster_num arrays that save the numeric features of all the dots 
                        that belong to some one cluster-category.
        cateIDSets: a list, it contains '%d' % cluster_num arrays that save the ID(i.e. the index in self.data or
                       self.name) of all the dots that belong to some one cluster-category.
        cateID: a list, len(self.clusterID) == self.dots. If you want to know the 5th dot's cluster-category, you
                   can key down self.clusterID[5].
        """
        self.dots = 0
        self.dims = 0
        self.data = []
        self.name = []

    def readFile(self,dataFilePath):
        self.data = np.loadtxt(dataFilePath,dtype=float)
        (self.dots,self.dims) = self.data.shape
        return

    def readArray(self,arr):
        self.data = np.array(arr,dtype=float)
        (self.dots,self.dims) = self.data.shape
        return

    def readNameFeatFile(self, path):
        """
        Parse the file that contains dot names and corresponding features.
        Given file's content is assumed to be written like this:
        
        ---------------FILE CONTENT REGION---------------
        dots_num(int) features_num(int)
        dot_name(str)
        dot_features(num)(with comma seperated)
        dot_name(str)
        dot_features(num)(with comma seperated)
        ...
        ...
        dot_name(str)
        dot_features(num)(with comma seperated)
        ---------------FILE CONTENT REGION---------------

        Example file:

        ---------------FILE CONTENT REGION---------------
        4096 3
        (10,20)
        1.2,3,10,
        (0,-8)
        -1,350,2,
        ...
        ...
        ---------------FILE CONTENT REGION---------------

        Parameters
        ----------
        path: str
            the path to get your file

        """
        handle = open(path, 'r')
        data = handle.read().split('\n')
        handle.close()
        dotsanddims = data[0].split(' ')
        self.dots = int(dotsanddims[0])
        self.dims = int(dotsanddims[1])
        data = data[1:]
        # get feat
        feats = np.empty((self.dots,self.dims),dtype=float)
        for i in range(self.dots):
            featstr = data[2*i+1].split(',')
            featnum = [float(featstr[k]) for k in range(self.dims)]
            feats[i] = featnum
        self.data = feats
        # get name
        names = []
        for i in range(self.dots):
            names.append(data[2*i])
        self.name = names
        return

    def zscore(self):
        """
        For the i-th feature-dim:
        mean(i) := the arithmetic mean of all dots' i-th feature.
        std(i)  := the standard variation of all dots' i-th feature.

        Converting Formula:
            new_feat(i) = (old_feat(i) - mean(i)) / std(i)
        """
        mean = self.data.mean(axis=0)
        self.mean = mean
        std = self.data.std(axis=0)
        self.std = std
        for index in range(self.dims):
            assert std[index] > 1E-20, "ERROR! The %d-th feature's std is too small!" % index
        self.data = (self.data - mean) / std



