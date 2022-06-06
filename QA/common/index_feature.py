from collections import OrderedDict
class IndexedFeature:
    def __init__(self):
        self.data = OrderedDict()

    def add(self, k, v=1.0):
        if k in self.data:
            self.data[k] = self.data[k] + v
        else:
            self.data[k] = v

    def add_if_absent(self, k, v=1.0):
        if k not in self.data:
            self.data[k] = v

    def add_set(self, other):
        for k, v in other.data.items():
            self.data[k] = v

    def __getitem__(self, k):
        return self.data.get(k, 0.)
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def add_prefix(self, prefix):
        new_data = OrderedDict([(prefix + k, v) for (k, v) in self.data.items()])
        self.data = new_data

class FeatureVocab:
    def __init__(self):
        self.feat_to_id = {}
        self.id_to_feat = {}

    def __getitem__(self, word):
        return self.feat_to_id.get(word, -1)

    def __contains__(self, word):
        return word in self.feat_to_id

    def __len__(self):
        return len(self.feat_to_id)

    def size(self):
        return len(self.feat_to_id)

    def get_word(self, wid):
        return self.id_to_feat[wid]

    def get_names(self):
        return [self.id_to_feat[i] for i in range(len(self))]
    def add(self, word):
        if word not in self:
            wid = self.feat_to_id[word] = len(self)
            self.id_to_feat[wid] = word
            return wid