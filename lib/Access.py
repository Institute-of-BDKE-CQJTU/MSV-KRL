from owlready2 import *


class Access(object):

    # 构造函数，获取需要映射本体的路径
    def __init__(self, ontology_file_path):
        self.ontology_file_path = ontology_file_path


    # 使用owlready2加载本体，然后再获取一个本体的graph对象
    def load_ontology(self):
        self.onto = get_ontology(self.ontology_file_path).load()

        self.graph = default_world.as_rdflib_graph()
        print("当前本体三元组数量为：{}\n".format(len(self.graph)))

    
    # 使用rdflib.graph的query方法查询相关信息
    def query_graph(self, sparql):
        results = self.graph.query(sparql)

        return list(results)
    

    # 使用owlready2对象获取当前本体的所有对象属性
    def get_object_properties(self):
        return self.onto.object_properties()
    

    def get_data_properties(self):
        return self.onto.data_properties()
    

    def get_annotation_properties(self):
        return self.onto.annotation_properties()
    

    def get_classes(self):
        return self.onto.classes()
    

    def get_individuals(self):
        return self.onto.individuals()