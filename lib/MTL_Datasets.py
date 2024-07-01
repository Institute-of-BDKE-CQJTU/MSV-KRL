import numpy as np
from lib.Helper import Helper
from rdflib.namespace import RDF, RDFS
from torch.utils.data import Dataset


class MTL_Train_Dataset(Dataset):
    ## 当前dataset，用于多任务协同训练，接收一个样本，同时返回正例和一个负例，即动态生成负例。
    ## 同时，也需要动态生成样本的标签

    def __init__(self, train_data, entity2idx=None, relation2idx=None):
        self.data = train_data

        if entity2idx != None and relation2idx != None:
            self.entity2idx = entity2idx
            self.relation2idx = relation2idx
        
        self.label_dict = {
            0: [0, 0, 0, 0, 0],
            Helper.CLASS_SUBSUMPTION: [1.0, 0.0, 0.0, 0.0, 0.0],
            Helper.CLASS_MEMBERSHIP: [0.0, 1.0, 0.0, 0.0, 0.0],
            Helper.CLASS_PROPERTY: [0.0, 0.0, 1.0, 0.0, 0.0],
            Helper.INDIVIDUAL_OBJECT_PROPERTY: [0.0, 0.0, 0.0, 1.0, 0.0],
            Helper.INDIVIDUAL_DATA_PROPERTY: [0.0, 0.0, 0.0, 0.0, 1.0]
        }


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        triple = self.data[index]
        triple_type = Helper.judge_triple_type(triple=triple)
        label = self.label_dict[triple_type]
        triple_idx = [self.entity2idx[triple[0]], self.relation2idx[triple[1]], self.entity2idx[triple[2]]]

        neg_triple = Helper.get_negative_sample(triple=triple)
        neg_label = self.label_dict[0]
        neg_triple_idx = [self.entity2idx[neg_triple[0]], self.relation2idx[neg_triple[1]], self.entity2idx[neg_triple[2]]]

        return [[triple_idx, label], [neg_triple_idx, neg_label]]
    



class MTL_Evaluate_Dataset(Dataset):
    def __init__(self, data, entity2idx=None, relation2idx=None, args=None):
        self.data = data
        self.args = args

        if entity2idx != None and relation2idx != None:
            self.entity2idx = entity2idx
            self.relation2idx = relation2idx


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        triple = self.data[index]
        triple_type = Helper.judge_triple_type(triple)

        candidate_entities, candidate_triples = Helper.get_triples_with_candidate_entity(triple)
                        
        candidate_entities_idx = list()
        for candidate_entity in candidate_entities:
            candidate_entities_idx.append(self.entity2idx[candidate_entity])

        candidate_triples_idx = list()
        for candidate_triple in candidate_triples:
            candidate_triple_idx = [self.entity2idx[candidate_triple[0]], self.relation2idx[candidate_triple[1]], self.entity2idx[candidate_triple[2]]]
            candidate_triples_idx.append(candidate_triple_idx)

        return np.array(candidate_triples_idx), np.array(self.entity2idx[triple[-1]]), np.array(candidate_entities_idx), triple_type
    

class MTL_Valid_Dataset(Dataset):
    def __init__(self, data, entity2idx=None, relation2idx=None):
        self.data = data

        if entity2idx != None and relation2idx != None:
            self.entity2idx = entity2idx
            self.relation2idx = relation2idx

        self.label_dict = {
            0: [0, 0, 0, 0, 0],
            Helper.CLASS_SUBSUMPTION: [1.0, 0.0, 0.0, 0.0, 0.0],
            Helper.CLASS_MEMBERSHIP: [0.0, 1.0, 0.0, 0.0, 0.0],
            Helper.CLASS_PROPERTY: [0.0, 0.0, 1.0, 0.0, 0.0],
            Helper.INDIVIDUAL_OBJECT_PROPERTY: [0.0, 0.0, 0.0, 1.0, 0.0],
            Helper.INDIVIDUAL_DATA_PROPERTY: [0.0, 0.0, 0.0, 0.0, 1.0]
        }


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        triple = self.data[index]
        triple_type = Helper.judge_triple_type(triple)

        label = self.label_dict[triple_type]
        triple_idx = [self.entity2idx[triple[0]], self.relation2idx[triple[1]], self.entity2idx[triple[2]]]

        return triple_idx, label