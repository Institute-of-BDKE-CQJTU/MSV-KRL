import rdflib
from rdflib.namespace import RDF, RDFS


class Subgraph(object):

    def __init__(self, rdf_file_path) -> None:
        self.graph = rdflib.Graph()
        self.graph.parse(rdf_file_path)

        self.sub_class_of_graph = rdflib.Graph()    # 类层次关系--subClassOf
        self.class_props_graph = rdflib.Graph()     # 类之间的对象属性和数据属性关系
        self.class_type_graph = rdflib.Graph()      # 实例的类型--type
        self.obj_props_graph = rdflib.Graph()       # 实例之间的对象属性
        self.data_props_graph = rdflib.Graph()      # 实例之间的数据属性


    def generate_subgraphs(self, classes, individuals, obj_props, data_props, subgraph_save_path):
        sub_class_of_uri = str(RDFS.subClassOf)
        type_uri = str(RDF.type)

        for s, p, o in self.graph:
            if str(p) == sub_class_of_uri:
                self.sub_class_of_graph.add((s, p, o))
            if (str(p) in obj_props and str(s) in classes and str(o) in classes) or (str(p) in data_props and str(s) in classes):
                self.class_props_graph.add((s, p, o))
            if str(p) == type_uri:
                self.class_type_graph.add((s, p, o))
            if str(p) in obj_props and str(s) in individuals and str(o) in individuals:
                self.obj_props_graph.add((s, p, o))
            if str(p) in data_props and str(s) in individuals:
                self.data_props_graph.add((s, p, o))

        # 便于人工查找问题
        self.sub_class_of_graph.serialize(subgraph_save_path+'subclassof.nt', format='nt')
        self.class_props_graph.serialize(subgraph_save_path+'class_props.nt', format='nt')
        self.class_type_graph.serialize(subgraph_save_path+'ind_type.nt', format='nt')
        self.obj_props_graph.serialize(subgraph_save_path+'obj_props.nt', format='nt')
        self.data_props_graph.serialize(subgraph_save_path+'data_props.nt', format='nt')

        # 用于数据处理
        self.sub_class_of_graph.serialize(subgraph_save_path+'subclassof.xml', format='xml')
        self.class_props_graph.serialize(subgraph_save_path+'class_props.xml', format='xml')
        self.class_type_graph.serialize(subgraph_save_path+'ind_type.xml', format='xml')
        self.obj_props_graph.serialize(subgraph_save_path+'obj_props.xml', format='xml')
        self.data_props_graph.serialize(subgraph_save_path+'data_props.xml', format='xml')


    