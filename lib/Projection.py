from lib.Access import Access
from lib.Annotations import Annotations
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, OWL


class Projection(object):

    def __init__(self, ontology_file_path):
        self.ontology_file_path = ontology_file_path
        self.propagate_domain_range = True

        self.access = Access(ontology_file_path)
        self.access.load_ontology()

        self.annotation_uris = Annotations()
        self.entity_to_labels_dict = dict()
        self.entity_to_synonyms_dict = dict()
        self.entity_to_all_annotations_dict = dict()

        self.obj_props = set()
        self.data_props = set()
        self.anno_props = set()
        self.classes = set()
        self.individuals = set()

        self.axioms_manchester = set()



    ####################
    #### 在生成URI文档的时候，同时需要添加公理信息，在这里将公理信息转换成Manchester语法
    ####################
    def create_manchester_syntax_axiom(self):
        print("开始将公理信息构造成Manchester语法\n")

        self.restriction = {
            24: "some",
            25: "only",
            26: "exactly",
            27: "min",
            28: "max"
        }

        ## Class axioms  ---子类、等价类
        for cls in self.access.get_classes():
            for cls_exp in cls.is_a:
                self.convert_axiom_to_manchester_syntax(cls.iri, cls_exp, "SubClassOf")
            for cls_exp in cls.equivalent_to:
                self.convert_axiom_to_manchester_syntax(cls.iri, cls_exp, "EquivalentTo")
        
        # Class assertions  ---类实例
        results = self.access.query_graph(self.get_query_for_all_class_types())
        for row in results:
            self.axioms_manchester.add(str(row[0]) + " Type " + str(row[1]))

        # Object Role assertions  ---对象属性
        for prop in list(self.access.get_object_properties()):
            results = self.access.query_graph(self.get_query_for_object_role_assertions(prop.iri))
            for row in results:
                self.axioms_manchester.add(str(row[0]) + " " + str(prop.iri) + " " + str(row[1]))

        # Data Role assertions  --数据属性
        for prop in list(self.access.get_data_properties()):
            results = self.access.query_graph(self.get_query_for_data_role_assertion(prop.iri))
            for row in results:
                self.axioms_manchester.add(str(row[0]) + " " + str(prop.iri) + " " + str(row[1]))

        print("公理Manchester语法构造结束\n")


    def convert_axiom_to_manchester_syntax(self, cls_iri, cls_exp, axiom_type):
        manchester_str = str(self.convert_expression_to_manchester_syntax(cls_exp))
        if manchester_str == "http://www.w3.org/2002/07/owl#Thing" or manchester_str == "http://www.w3.org/2002/07/owl#Nothing":
            return

        self.axioms_manchester.add(str(cls_iri) + " " + axiom_type + " " + manchester_str)


    def convert_union_to_manchester_syntax(self, cls_exp):
        return self.convert_list_to_manchester_syntax(cls_exp, "or")


    def convert_intersection_to_manchester_syntax(self, cls_exp):
        return self.convert_list_to_manchester_syntax(cls_exp, "and")


    def convert_list_to_manchester_syntax(self, cls_exp, connector):
        i = 0
        manchester_str = ""
        while i < len(cls_exp.Classes)-1:
            #print(cls_exp.Classes[i])
            manchester_str += self.convert_expression_to_manchester_syntax(cls_exp.Classes[i]) + " " + connector + " "
            i += 1

        return manchester_str + self.convert_expression_to_manchester_syntax(cls_exp.Classes[i])


    def convert_restriction_to_manchester_syntax(self, cls_exp):
        if hasattr(cls_exp.property, "iri"):
            manchester_str = str(cls_exp.property.iri)
        else:  ## case of inverses
            manchester_str = str(cls_exp.property)

        manchester_str += " " + self.restriction[cls_exp.type]

        if cls_exp.type >= 26:
            manchester_str += " " + str(cls_exp.cardinality)

        return manchester_str + " " + self.convert_expression_to_manchester_syntax(cls_exp.value)


    def convert_atomic_class_to_manchester_syntax(self, cls_exp):
        return str(cls_exp.iri)


    def convert_one_of_to_manchester_syntax(self, cls_exp):
        i = 0
        manchester_str = "OneOf: "
        while i < len(cls_exp.instances)-1:
            #print(cls_exp.Classes[i])
            manchester_str += self.convert_expression_to_manchester_syntax(cls_exp.instances[i]) + ", "
            i += 1

        return manchester_str + self.convert_expression_to_manchester_syntax(cls_exp.instances[i])


    def convert_expression_to_manchester_syntax(self, cls_exp):
        try:
            #Union or Intersection
            if hasattr(cls_exp, "Classes"):
                if hasattr(cls_exp, "get_is_a"):
                    return self.convert_intersection_to_manchester_syntax(cls_exp)
                else:
                    return self.convert_union_to_manchester_syntax(cls_exp)

            #Restriction
            elif hasattr(cls_exp, "property"):
                return self.convert_restriction_to_manchester_syntax(cls_exp)

            #Atomic class
            elif hasattr(cls_exp, "iri"):
                return self.convert_atomic_class_to_manchester_syntax(cls_exp)

            #One of
            elif hasattr(cls_exp, "instances"):
                return self.convert_one_of_to_manchester_syntax(cls_exp)

            else: ##Any other expression (e.g., a datatype)
                return str(cls_exp)
            
        except:# AttributeError:
            return str(cls_exp)  # In case of error
        
    ####################
    #### 公理Manchester语法生成结束
    ####################



    ## 获取类、实例、对象属性、数据属性的iri
    def extract_uris(self):
        for obj_prop in self.access.get_object_properties():
            self.obj_props.add(obj_prop.iri)

        for data_prop in self.access.get_data_properties():
            self.data_props.add(data_prop.iri)

        for anno_prop in self.access.get_annotation_properties():
            self.anno_props.add(anno_prop.iri)

        for cls in self.access.get_classes():
            self.classes.add(cls.iri)

        for individual in self.access.get_individuals():
            self.individuals.add(individual.iri)

    def get_obj_props(self):
        return self.obj_props
    
    def get_data_props(self):
        return self.data_props
    
    def get_anno_props(self):
        return self.anno_props
    
    def get_classes(self):
        return self.classes
    
    def get_individuals(self):
        return self.individuals



    ####################
    #### 将本体映射为RDF图
    ####################
    def extract_and_projection(self):
        print("开始将本体映射为RDF图...\n")

        ## 初始化一个rdflib的Graph对象，用于存储抽取出来的本体
        self.graph = Graph()
        self.graph.bind("owl", "http://www.w3.org/2002/07/owl#")
        self.graph.bind("skos", "http://www.w3.org/2004/02/skos/core#")
        # self.graph.bind("obo1", "http://www.geneontology.org/formats/oboInOwl#")  
        # self.graph.bind("obo2", "http://www.geneontology.org/formats/oboInOWL#")  


        ####################
        ## 一、抽取最简单的、最常用的三元组
        ## 1、抽取子类三元组
        print("开始抽取子类三元组...")
        results = self.access.query_graph(self.get_query_for_atomic_class_subsumptions())
        for row in results:
            self.add_subsumption_triple(row[0], row[1])
        print("子类三元组抽取结束，共抽取{}个三元组\n".format(len(results)))

        ## 2、抽取类实例三元组
        print("开始抽取类实例三元组...")
        results = self.access.query_graph(self.get_query_for_all_class_types())
        for row in results:
            self.add_class_type_triple(row[0], row[1])
        print("类实例三元组抽取结束，共抽取{}个三元组\n".format(len(results)))

        ## 3、抽取同一性（等价）实例三元组（sameAs，表示两个实例是等价的）
        print("开始抽取等价实例三元组")
        results = self.access.query_graph(self.get_query_for_all_same_as())
        for row in results:
            self.add_same_as_triple(row[0], row[1])
        print("等价实例三元组抽取结束，共抽取{}个三元组\n".format(len(results)))

        ## 简单三元组抽取结束
        ####################


        ####################
        ## 二、抽取对象属性
        self.triple_dict = {}

        self.domains = set()
        self.ranges = set()

        self.domains_dict = {}
        self.ranges_dict = {}

        print("开始抽取对象属性三元组")
        for property in list(self.access.get_object_properties()):
            print("\t开始抽取对象属性{}".format(property.iri))

            self.domains_dict[property.iri] = set()
            self.ranges_dict[property.iri] = set()

            self.triple_dict.clear()
            self.domains.clear()
            self.ranges.clear()
            property_triple_count = 0

            ## 1、简单的domain和range
            results = self.access.query_graph(self.get_query_for_domain_and_range(property.iri))
            print("\t\t简单domain和range三元组个数为：{}".format(len(results)))     
            property_triple_count += len(results)
            self.process_property_results(property.iri, results, True, True)

            results_domain = self.access.query_graph(self.get_query_for_domain(property.iri))
            results_range = self.access.query_graph(self.get_query_for_range(property.iri))
            for row_domain in results_domain:
                self.domains.add(row_domain[0])
                self.domains_dict[property.iri].add(row_domain[0])
            for row_range in results_range:
                self.ranges.add(row_range[0])
                self.ranges_dict[property.iri].add(row_range[0])


            ###############################目前还有点问题的就是这个地方，第2点##############################################
            ## 2.1、复杂的domain和range
            results_domain = self.access.query_graph(self.get_query_for_complex_domain(property.iri))
            results_range = self.access.query_graph(self.get_query_for_complex_range(property.iri))

            if len(results_domain) != 0 and len(results_range) == 0:
                results_range = self.access.query_graph(self.get_query_for_range(property.iri))
            if len(results_domain) == 0 and len(results_range) != 0:
                results_domain = self.access.query_graph(self.get_query_for_domain(property.iri))
            print("\t\t复杂domain和range三元组个数为：{}".format(len(results_domain) * len(results_range)))
            property_triple_count += len(results_domain) * len(results_range)

            for row_domain in results_domain:

                for row_range in results_range:
                    self.add_triple(row_domain[0], URIRef(property.iri), row_range[0])
                    if not row_domain[0] in self.triple_dict:
                        self.triple_dict[row_domain[0]]=set()
                    self.triple_dict[row_domain[0]].add(row_range[0])
            
            ## 3、抽取属性的约束条件
            # 3.1 抽取简单的RHS属性约束
            # 3.1a
            results = self.access.query_graph(self.get_query_for_restrictions_RHS_subClassOf(property.iri))
            print("\t\t简单RHS subClassOf三元组个数为：{}".format(len(results)))
            property_triple_count += len(results)
            self.process_property_results(property.iri, results, True, True)


            # 3.1b
            results = self.access.query_graph(self.get_query_for_restrictions_RHS_equivalent_class(property.iri))
            print("\t\t简单RHS equivalentClass三元组个数为：{}".format(len(results)))
            property_triple_count += len(results)
            self.process_property_results(property.iri, results, True, True)
            

            # 3.3 抽取简单的LHS属性约束
            results = self.access.query_graph(self.get_query_for_restrictions_LHS_subClassOf(property.iri))
            print("\t\t简单LHS subClassOf三元组个数为：{}".format(len(results)))
            property_triple_count += len(results)
            self.process_property_results(property.iri, results, True, True)


            # 3.4 抽取复杂的LHS属性约束
            results = self.access.query_graph(self.get_query_for_complex_restrictions_LHS_subClassOf(property.iri))
            print("\t\t复杂LHS subClassOf三元组个数为：{}".format(len(results)))
            property_triple_count += len(results)
            self.process_property_results(property.iri, results, True, True)


            # 3.5 抽取最直接的对象属性
            results = self.access.query_graph(self.get_query_for_object_role_assertions(property.iri))
            print("\t\trole assertion三元组个数为：{}".format(len(results)))
            property_triple_count += len(results)
            self.process_property_results(property.iri, results, False, True)


            # 4、抽取当前对象属性的逆关系
            results = self.access.query_graph(self.get_query_for_inverses(property.iri))
            print("\t\t逆关系三元组个数为：{}".format(len(results) * property_triple_count))
            temp_triple_count = property_triple_count
            property_triple_count += len(results)
            for row in results:
                for subject in self.triple_dict:
                    for obj in self.triple_dict[subject]:
                        self.add_triple(obj, row[0], subject)

            # 5、抽取当前对象属性的等价关系
            results = self.access.query_graph(self.get_query_for_atomic_equivalent_object_properties(property.iri))
            print("\t\t等价关系三元组个数为：{}".format(len(results) * temp_triple_count))
            property_triple_count += len(results) * temp_triple_count
            for row in results:
                for subject in self.triple_dict:
                    for obj in self.triple_dict[subject]:
                        self.add_triple(subject, row[0], obj)


            print("\t抽取结束，共抽取{}个三元组\n".format(property_triple_count))
        print("对象属性抽取结束\n")

        ## 对象属性抽取结束
        ####################


        ####################
        ## 三、抽取数据属性
        print("开始抽取数据属性：")
        for data_prop in list(self.access.get_data_properties()):
            print("\t开始抽取数据属性{}".format(data_prop.iri))

            data_prop_triple_count = 0

            self.domains_dict[data_prop.iri] = set()    # 复杂推理

            self.triple_dict.clear()
            # 简单推理
            self.domains.clear()
            self.ranges.clear()     

            ## 1、抽取数据当前数据属性的domain
            results_domain = self.access.query_graph(self.get_query_for_domain(data_prop.iri))
            for row_domain in results_domain:
                self.domains.add(row_domain[0])
                self.domains_dict[data_prop.iri].add(row_domain[0])

            ## 1a、新添加内容：添加数据属性的domain和range，range一般是datatype且只有一个，而domain可能包含多个或者一个
            results_domain = self.access.query_graph(self.get_query_for_data_prop_complex_domain(data_prop.iri))
            results_range = self.access.query_graph(self.get_query_for_range(data_prop.iri))
            for row_domain in results_domain:
                for row_range in results_range:
                    self.add_triple(row_domain[0], URIRef(data_prop.iri), row_range[0])
                    if not row_domain[0] in self.triple_dict:
                        self.triple_dict[row_domain[0]]=set()
                    self.triple_dict[row_domain[0]].add(row_range[0])

            ## 2、抽取属性约束
            # 2.1 RHS subClassOf
            results = self.access.query_graph(self.get_query_for_data_restriction_RHS_subClassOf(data_prop.iri))
            self.process_property_results(data_prop.iri, results, True, False)  
            # 2.2 RHS equivalentClass
            results = self.access.query_graph(self.get_query_for_data_restriction_RHS_equivalentClass(data_prop.iri))
            self.process_property_results(data_prop.iri, results, True, False)

            ## 3、抽取数据属性的角色声明（role assertion）
            results = self.access.query_graph(self.get_query_for_data_role_assertion(data_prop.iri))
            data_prop_triple_count += len(results)
            print("\t\t抽取数据属性角色声明共{}三元组".format(len(results)))
            self.process_property_results(data_prop.iri, results, False, True)

            ## 4、抽取当前数据属性的等价属性【注：数据属性不应存在逆属性】
            results = self.access.query_graph(self.get_query_for_atomic_equivalent_data_prop(data_prop.iri))
            for row in results:
                for sub in self.triple_dict:
                    for obj in self.triple_dict[sub]:
                        self.add_triple(sub, row[0], obj)
            print("\t该数据属性抽取结束，共{}三元组\n".format(data_prop_triple_count*len(results) + data_prop_triple_count))

        print("所有数据属性抽取结束\n")

        ## 对象属性抽取结束
        ####################

        ####################
        ## 5、复杂公理抽取

        print("开始抽取复杂公理，包括等价、子类")
        self.extract_triples_from_complex_axiom()
        print("复杂公理抽取结束\n")         

        ## 复杂公理抽取结束
        ###################


        ####################
        ## 四、抽取标注属性
        print("开始抽取标注属性：")
        all_annotation_uris = self.annotation_uris.get_lexical_annotation_iris()

        for anno_prop_uri in all_annotation_uris:
            print("\t开始抽取数据属性{}".format(anno_prop_uri))
            results = self.access.query_graph(self.get_query_for_annotations(anno_prop_uri))

            anno_prop_triple_count = 0

            for row in results:
                try:
                    if row[1].language == 'en' or row[1].language == None:
                        anno_prop_triple_count += 1
                        self.add_triple(row[0], URIRef(anno_prop_uri), row[1])
                except AttributeError:
                    pass
            print("\t当前标注属性共抽取{}三元组\n".format(anno_prop_triple_count))

        print("标注属性抽取结束\n")

        ## 标注属性抽取结束
        ####################

        print("RDF图已经映射完成\n")
    ## 所有三元组抽取结束
    ####################
        


    ####################
    #### 下面这些函数用于获取查询桥梁信息的SPARQL语句
    ####################
    # 获取查询桥梁子类关系的SPARQL语句
    def get_query_for_atomic_class_subsumptions(self):
        sparql = """SELECT ?s ?o WHERE { ?s <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?o .
        FILTER (isIRI(?s) && isIRI(?o)
        && str(?o) != 'http://www.w3.org/2002/07/owl#Nothing'
        && str(?s) != 'http://www.w3.org/2002/07/owl#Nothing'
        && str(?o) != 'http://www.w3.org/2002/07/owl#Thing'
        && str(?s) != 'http://www.w3.org/2002/07/owl#Thing'
        )
        }"""

        return sparql

    def get_query_for_all_class_types(self):
        sparql = """SELECT ?s ?o WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?o .
        FILTER (isIRI(?s) && isIRI(?o)
        && str(?o) != 'http://www.w3.org/2002/07/owl#Ontology'
        && str(?o) != 'http://www.w3.org/2002/07/owl#AnnotationProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#ObjectProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#Class'
        && str(?o) != 'http://www.w3.org/2002/07/owl#DatatypeProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#Restriction'
        && str(?o) != 'http://www.w3.org/2002/07/owl#NamedIndividual'
        && str(?o) != 'http://www.w3.org/2002/07/owl#Thing'
        && str(?o) != 'http://www.w3.org/2002/07/owl#TransitiveProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#FunctionalProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#InverseFunctionalProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#SymmetricProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#AsymmetricProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#ReflexiveProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#IrreflexiveProperty'
        )
        }"""

        return sparql
    
    def get_query_for_all_same_as(self):
        sparql = """SELECT ?s ?o WHERE { ?s <http://www.w3.org/2002/07/owl#sameAs> ?o .
        filter( isIRI(?s) && isIRI(?o))
        }"""

        return sparql
    
    def get_query_for_domain(self, property_iri):
        sparql = """SELECT DISTINCT ?d WHERE {{ <{prop}> <http://www.w3.org/2000/01/rdf-schema#domain> ?d .
        FILTER (isIRI(?d))
        }}""".format(prop=property_iri)

        return sparql
    
    def get_query_for_range(self, property_iri):
        sparql = """SELECT DISTINCT ?r WHERE {{ <{prop}> <http://www.w3.org/2000/01/rdf-schema#range> ?r .
        FILTER (isIRI(?r))
        }}""".format(prop=property_iri)

        return sparql
    
    def get_query_for_domain_and_range(self, property_iri):
        sparql = """SELECT DISTINCT ?d ?r WHERE {{ <{prop}> <http://www.w3.org/2000/01/rdf-schema#domain> ?d .
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#range> ?r .
        FILTER (isIRI(?d) && isIRI(?r))
        }}""".format(prop=property_iri)

        return sparql
    
    def get_query_for_complex_domain(self, property_iri):
        sparqal = """SELECT DISTINCT ?d where {{
        {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#domain> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?d ] ] ] .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#domain> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?d ] ] ] .
        }}
        filter( isIRI( ?d ) )
        }}""".format(prop=property_iri)

        return sparqal
    
    def get_query_for_data_prop_complex_domain(self, property_iri):
        sparqal = """SELECT DISTINCT ?d where {{
        {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#domain> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?d ] ] ] .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#domain> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?d ] ] ] .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#domain> ?d .
        }}
        filter( isIRI( ?d ) )
        }}""".format(prop=property_iri)

        return sparqal

    def get_query_for_complex_range(self, property_iri):
        sparqal = """SELECT DISTINCT ?r where {{
        {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#range> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?r ] ] ] .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#range> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?r ] ] ] .
        }}
        filter( isIRI( ?r ) )
        }}""".format(prop=property_iri)

        return sparqal
    
    def get_query_for_restrictions_RHS_subClassOf(self, property_iri):
        sparql = """SELECT DISTINCT ?s ?o WHERE {{
        ?s <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?bn .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> ?o .
        }}
        FILTER (isIRI(?s) && isIRI(?o))
        }}""".format(prop=property_iri)

        return sparql

    def get_query_for_restrictions_RHS_equivalent_class(self, property_iri):
        sparql = """SELECT DISTINCT ?s ?o WHERE {{
        ?s <http://www.w3.org/2002/07/owl#equivalentClass> ?bn .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> ?o .
        }}
        FILTER (isIRI(?s) && isIRI(?o))
        }}""".format(prop=property_iri)

        return sparql

    def get_query_for_complex_restrictions_RHS_subClassOf(self, property_iri):
        sparql = """SELECT DISTINCT ?s ?o WHERE {{
        ?s <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?bn .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        FILTER (isIRI(?s) && isIRI(?o))
        }}""".format(prop=property_iri)

        return sparql

    def get_query_for_complex_restrictions_RHS_equivalent_class(self, property_iri):
        sparql = """SELECT DISTINCT ?s ?o WHERE {{
        ?s <http://www.w3.org/2002/07/owl#equivalentClass> ?bn .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        FILTER (isIRI(?s) && isIRI(?o))
        }}""".format(prop=property_iri)

        return sparql

    def get_query_for_restrictions_LHS_subClassOf(self, property_iri):
        sparql = """SELECT DISTINCT ?s ?o WHERE {{
        ?bn <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?s .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> ?o .
        }}
        FILTER (isIRI(?s) && isIRI(?o))
        }}""".format(prop=property_iri)

        return sparql

    def get_query_for_complex_restrictions_LHS_subClassOf(self, property_iri):
        sparql = """SELECT DISTINCT ?s ?o WHERE {{
        ?bn <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?s .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        FILTER (isIRI(?s) && isIRI(?o))
        }}""".format(prop=property_iri)

        return sparql

    def get_query_for_object_role_assertions(self, property_iri):
        sparql = """SELECT ?s ?o WHERE {{ ?s <{prop}> ?o .
        filter( isIRI(?s) && isIRI(?o) )
        }}""".format(prop=property_iri)

        return sparql 
    
    def get_query_for_inverses(self, property_iri):
        sparql = """SELECT DISTINCT ?p WHERE {{
        {{
        ?p <http://www.w3.org/2002/07/owl#inverseOf> <{prop}> .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2002/07/owl#inverseOf> ?p .
        }}
        filter(isIRI(?p))
        }}""".format(prop=property_iri)

        return sparql

    def get_query_for_atomic_equivalent_object_properties(self, property_iri):
        sparql = """SELECT DISTINCT ?p WHERE {{
        {{
        ?p <http://www.w3.org/2002/07/owl#equivalentProperty> <{prop}> .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2002/07/owl#equivalentProperty> ?p .
        }}
        FILTER (isIRI(?p))
        }}""".format(prop=property_iri)

        return sparql
    
    def get_query_for_data_restriction_RHS_subClassOf(self, property_iri):
        sparql = """SELECT DISTINCT ?s WHERE {{
        ?s <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?bn .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        FILTER (isIRI(?s))
        }}""".format(prop=property_iri)

        return sparql
    
    def get_query_for_data_restriction_RHS_equivalentClass(self, property_iri):
        sparql = """SELECT DISTINCT ?s WHERE {{
        ?s <http://www.w3.org/2002/07/owl#equivalentClass> ?bn .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        FILTER (isIRI(?s))
        }}""".format(prop=property_iri)

        return sparql
    
    def get_query_for_data_role_assertion(self, property_iri):
        sparql = """SELECT ?s ?o WHERE {{ ?s <{prop}> ?o .
        filter( isIRI(?s) )
        }}""".format(prop=property_iri)

        return sparql
    
    def get_query_for_atomic_equivalent_data_prop(self, property_iri):
        sparql = """SELECT DISTINCT ?p WHERE {{
        {{
        ?p <http://www.w3.org/2002/07/owl#equivalentProperty> <{prop}> .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2002/07/owl#equivalentProperty> ?p .
        }}
        FILTER (isIRI(?p))
        }}""".format(prop=property_iri)

        return sparql
    
    def get_query_for_annotations(self, anno_prop_iri):
        sparql = """SELECT DISTINCT ?s ?o WHERE {{
        {{
        ?s <{ann_prop}> ?o .
        }}
        UNION
        {{
        ?s <{ann_prop}> ?i .
        ?i <http://www.w3.org/2000/01/rdf-schema#label> ?o .
        }}
        }}""".format(ann_prop=anno_prop_iri)

        return sparql
    


    ####################
    #### 对属性的domain和range进行模糊推理
    ####################
    def process_property_results(self, prop_iri, results, are_tbox_results, add_triple):
        for row in results:
            if add_triple:
                self.add_triple(row[0], URIRef(prop_iri), row[1])
                if not row[0] in self.triple_dict:
                    self.triple_dict[row[0]] = set()
                self.triple_dict[row[0]].add(row[1])

            if self.propagate_domain_range:
                if are_tbox_results:
                    self.propagate_tbox_domain(row[0])
                    try:
                        self.propagate_tbox_range(row[1])
                    except:
                        pass
                else:
                    self.propagate_abox_domain(row[0])
                    try:
                        self.propagate_abox_range(row[1])
                    except:
                        pass

    def propagate_tbox_domain(self, source):
        for domain in self.domains:
            if str(source) == str(domain):
                continue

            self.add_subsumption_triple(source, domain) 

    def propagate_tbox_range(self, target):
        for rangi in self.ranges:
            if str(target) == str(rangi):
                continue

            self.add_subsumption_triple(target, rangi)
    
    def propagate_abox_domain(self, source):
        for domain in self.domains:
            self.add_class_type_triple(source, domain)

    def propagate_abox_range(self, target):
        for rangi in self.ranges:
            self.add_class_type_triple(target, rangi)
    


    ####################
    #### 下面这些函数用于抽取复杂公理
    ####################
    def extract_triples_from_complex_axiom(self):
        ## A sub/equiv B and/or R some D
        ## A sub/equiv R some (D and/or E)

        for cls in self.access.get_classes():

            expressions = set()
            expressions.update(cls.is_a, cls.equivalent_to)

            
            for cls_exp in expressions:

                try:

                    for cls_exp2 in cls_exp.Classes:  ##Accessing the list in Classes
                        
                        try:              
                            self.add_subsumption_triple(URIRef(cls.iri), URIRef(cls_exp2.iri))

                        except AttributeError:
                            try:
                                self.extract_triples_for_restriction(cls, cls_exp2)

                            except AttributeError:
                                pass  

                except AttributeError:
                    try:
                        self.extract_triples_for_restriction(cls, cls_exp)

                    except AttributeError:
                        pass  

    def extract_triples_for_restriction(self, cls, cls_exp_rest):
        try:
            targets = set()

            property_iri = cls_exp_rest.property.iri

            if self.propagate_domain_range:
                if property_iri in self.domains_dict:
                    for domain_cls in self.domains_dict[property_iri]:
                        if str(cls.iri) == str(domain_cls):
                            continue

                        self.add_subsumption_triple(URIRef(cls.iri), domain_cls)

            if hasattr(cls_exp_rest.value, "Classes"):
                for target_cls in cls_exp_rest.value.Classes:

                    if hasattr(target_cls, "iri"):  
                        targets.add(target_cls.iri)
            #Atomic target class in target of restrictions
            elif hasattr(cls_exp_rest.value, "iri"):
                ##Error with reviewsPerPaper exactly 1 rdfs:Literal restriction
                ##rdfs:Literal is considered as owl:Thing
                #In any case both rdfs:Literal and owl:Thing should be filtered
                target_cls_iri = cls_exp_rest.value.iri
                
                if not target_cls_iri=="http://www.w3.org/2002/07/owl#Thing" and not target_cls_iri=="http://www.w3.org/2000/01/rdf-schema#Literal":
                    targets.add(target_cls_iri)

                    #TODO Propagate range only in this case
                    if self.propagate_domain_range:
                        #In case of unexpected cases
                        if property_iri in self.ranges_dict:
                            for range_cls in self.ranges_dict[property_iri]:
                                if str(target_cls_iri) == str(range_cls):
                                    continue

                                self.add_subsumption_triple(URIRef(target_cls_iri), range_cls)

            ##end creation of targets
            for target_cls in targets:
                self.add_triple(URIRef(cls.iri), URIRef(property_iri), URIRef(target_cls))

                ##Propagate equivalences and inverses for cls_exp2.property
                ## 12a. Extract named inverses and create/propagate new reversed triples.
                results = self.access.query_graph(self.get_query_for_inverses(property_iri))
                for row in results:
                    self.add_triple(URIRef(target_cls), row[0], URIRef(cls.iri)) #Reversed triple. Already all as URIRef

                ## 12b. Propagate property equivalences only (object).
                results = self.access.query_graph(self.get_query_for_atomic_equivalent_object_properties(property_iri))
                for row in results:
                    self.add_triple(URIRef(cls.iri), row[0], URIRef(target_cls)) #Inferred triple. Already all as URIRef
            ##end targets

        except AttributeError:
            pass  



    ####################
    #### 下面这些函数用于将获取到的相关信息构造为三元组添加进初始化的rdflib的graph对象当中
    ####################
    def add_subsumption_triple(self, subject_iri, object_iri):
        triple = (subject_iri, RDFS.subClassOf, object_iri)
        self.graph.add(triple)

    def add_class_type_triple(self, subject_iri, object_iri):
        triple = (subject_iri, RDF.type, object_iri)
        self.graph.add(triple)

    def add_same_as_triple(self, subject_iri, object_iri):
        triple = (subject_iri, OWL.sameAs, object_iri)
        self.graph.add(triple)

    def add_triple(self, subject_iri, predicate_iri, object_iri):
        triple = (subject_iri, predicate_iri, object_iri)
        self.graph.add(triple)


    def save_rdf_graph_to_file(self, rdf_file_dir, ontology_name):
        self.graph.serialize(destination=rdf_file_dir+ontology_name+'.xml', format='xml')
        self.graph.serialize(destination=rdf_file_dir+ontology_name+'.nt', format='nt')

        print("RDF文件保存到'{}'，格式为RDF/XML、N-Triple，共{}个三元组\n".format(rdf_file_dir, len(self.graph)))




    ####################
    #### 下面这些函数用于获取标注信息
    ####################
    def get_annotations(self):
        # 获取所有的基本的标注属性iri
        pref_label_annotation_uris = set()
        pref_label_annotation_uris.update(self.annotation_uris.get_prefered_label_annotation_iris())

        synonyms_annotation_uris = set()
        synonyms_annotation_uris.update(self.annotation_uris.get_synonyms_annotation_iris())

        all_annotation_uris = set()
        all_annotation_uris.update(self.annotation_uris.get_lexical_annotation_iris())

        ## 接下来就是实现获取标注属性，并保存到字典里面
        self.query_annotations(pref_label_annotation_uris, self.entity_to_labels_dict)
        self.query_annotations(synonyms_annotation_uris, self.entity_to_synonyms_dict)
        self.query_annotations(all_annotation_uris, self.entity_to_all_annotations_dict)

        # print()


    def query_annotations(self, annotation_uris, dictionary):
        for anno_uri in annotation_uris:
            results = self.access.query_graph(self.get_query_for_annotations(anno_uri))
            for row in results:
                #Filter by language
                try:
                    if row[1].language=="en" or row[1].language==None:

                        if not str(row[0]) in dictionary:
                            dictionary[str(row[0])]=set()

                        dictionary[str(row[0])].add(row[1].value)       

                except AttributeError:
                    pass

    def get_labels_by_entity_uri(self, entity_uri):
        return self.entity_to_labels_dict[entity_uri]   


    def get_synonyms_by_entity_uri(self, entity_uri):
        return self.entity_to_synonyms_dict[entity_uri]


    def get_all_annotation_by_entity_uri(self, entity_uri):
        return self.entity_to_all_annotations_dict[entity_uri]



