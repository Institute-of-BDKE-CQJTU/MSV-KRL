import rdflib
from rdflib import URIRef
from lib.Random import RandomWalker
from lib.Graph import KnowledgeGraph, Vertex




class Random_Walk(object):
    def __init__(self, rdf_file_path, walk_depth, classes) -> None:
        self.classes = classes
        self.walk_depth = walk_depth

        self.graph = rdflib.Graph()
        if rdf_file_path.endswith('ttl') or rdf_file_path.endswith('TTL'):
            self.graph.parse(rdf_file_path, format='turtle')
        else:
            self.graph.parse(rdf_file_path)


    def get_walks(self):
        kg, walker = self.construct_kg_walker()
        instances = [URIRef(c) for c in self.classes]
        walks = list(walker.extract(graph=kg, instances=instances))

        return walks


    def construct_kg_walker(self):
        kg = KnowledgeGraph()
        for (s, p, o) in self.graph:
            s_v, o_v = Vertex(str(s)), Vertex(str(o))
            p_v = Vertex(str(p), predicate=True, _from=s_v, _to=o_v)
            kg.add_vertex(s_v)
            kg.add_vertex(p_v)
            kg.add_vertex(o_v)
            kg.add_edge(s_v, p_v)
            kg.add_edge(p_v, o_v)

        walker = RandomWalker(depth=self.walk_depth, walks_per_graph=float('inf'))

        return kg, walker
        