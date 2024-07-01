

class Annotations(object):
    
    def __init__(self):
        
        self.main_label_uris = set()
        self.synonym_label_uris = set()
        self.lexical_annotation_uris = set()
        
        # 通用label标注属性，几乎每一个本体都有
        self.main_label_uris.add("http://www.w3.org/2000/01/rdf-schema#label")
        # skos首选标签，通用性同上
        self.main_label_uris.add("http://www.w3.org/2004/02/skos/core#prefLabel")
        # 类似于这种IRI的都是属于OBO Foundry统一定义的，在生物医学本体领域可以通用。当前这个IRI表示“has definition”
        self.main_label_uris.add("http://purl.obolibrary.org/obo/IAO_0000111")
        # 这个IRI表示了OBI本体中的一个名为"has_specified_input"的属性，该属性用于描述某个研究实验或过程的指定输入。
        self.main_label_uris.add("http://purl.obolibrary.org/obo/IAO_0000589")


        # 同义词标注标签或者备选标签，除了一些显而易见的IRI外，其他的基本都有注释
        # 与当前实体具有相关含义的词或短语
        self.synonym_label_uris.add("http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym")
        # 与当前实体具有确切一致的词或短语
        self.synonym_label_uris.add("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym")
        # 对于synonym这个标注属性还是比较通用的
        self.synonym_label_uris.add("http://purl.bioontology.org/ontology/SYN#synonym")
        self.synonym_label_uris.add("http://scai.fraunhofer.de/CSEO#Synonym")
        self.synonym_label_uris.add("http://purl.obolibrary.org/obo/synonym")
        # 美国国家癌症研究所本体中的同义术语
        self.synonym_label_uris.add("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#FULL_SYN")
        # Experimental Factor Ontology
        self.synonym_label_uris.add("http://www.ebi.ac.uk/efo/alternative_term")
        # 同FULL_SYN
        self.synonym_label_uris.add("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#Synonym")
        self.synonym_label_uris.add("http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#Synonym")
        # has definition标注属性为什么会出现在这里【现在明白了，是因为实体的定义和实体本身具有相同含义】
        self.synonym_label_uris.add("http://www.geneontology.org/formats/oboInOwl#hasDefinition")
        self.synonym_label_uris.add("http://bioontology.org/projects/ontologies/birnlex#preferred_label")
        self.synonym_label_uris.add("http://bioontology.org/projects/ontologies/birnlex#synonyms")
        # skos中的备选标签和前面的preflabel对应
        self.synonym_label_uris.add("http://www.w3.org/2004/02/skos/core#altLabel")
        # 表示生态毒理学数据库中生物种的拉丁学名
        self.synonym_label_uris.add("https://cfpub.epa.gov/ecotox#latinName")
        # 用于表示生态毒理学数据库中生物种的通用名称
        self.synonym_label_uris.add("https://cfpub.epa.gov/ecotox#commonName")
        # 用于表示生物种的科学学名
        self.synonym_label_uris.add("https://www.ncbi.nlm.nih.gov/taxonomy#scientific_name")
        self.synonym_label_uris.add("https://www.ncbi.nlm.nih.gov/taxonomy#synonym")
        # equivalent name和synonym同义
        self.synonym_label_uris.add("https://www.ncbi.nlm.nih.gov/taxonomy#equivalent_name")
        self.synonym_label_uris.add("https://www.ncbi.nlm.nih.gov/taxonomy#genbank_synonym")
        self.synonym_label_uris.add("https://www.ncbi.nlm.nih.gov/taxonomy#common_name")       
        # 这个IRI代表了 OBI 本体中的一个名为 "has_specified_output" 的属性，表示一个实验或研究活动具有特定的输出。      
        self.synonym_label_uris.add("http://purl.obolibrary.org/obo/IAO_0000118") 
        
        #Lexically rich interesting
        self.lexical_annotation_uris.update(self.main_label_uris)
        self.lexical_annotation_uris.update(self.synonym_label_uris)
        
        self.lexical_annotation_uris.add("http://www.w3.org/2000/01/rdf-schema#comment")
        # 用于表示 Gene Ontology (GO) 词汇项与外部数据库引用之间的关联
        self.lexical_annotation_uris.add("http://www.geneontology.org/formats/oboInOwl#hasDbXref")

        # 都是表述描述信息
        self.lexical_annotation_uris.add("http://purl.org/dc/elements/1.1/description")
        self.lexical_annotation_uris.add("http://purl.org/dc/terms/description")
        self.lexical_annotation_uris.add("http://purl.org/dc/elements/1.1/title")
        self.lexical_annotation_uris.add("http://purl.org/dc/terms/title")
        
        # 表示官方定义
        self.lexical_annotation_uris.add("http://purl.obolibrary.org/obo/IAO_0000115")
        
        # 阐明、解释
        self.lexical_annotation_uris.add("http://purl.obolibrary.org/obo/IAO_0000600")
        
        
        # 有一阶逻辑的相关公理
        self.lexical_annotation_uris.add("http://purl.obolibrary.org/obo/IAO_0000602")
        # 有自然语言公理化的相关公理
        self.lexical_annotation_uris.add("http://purl.obolibrary.org/obo/IAO_0000601")
        # 用于表示 Gene Ontology (GO) 词汇项所属的命名空间，以支持对 GO 数据的组织和查询
        self.lexical_annotation_uris.add("http://www.geneontology.org/formats/oboInOwl#hasOBONamespace")

        # HeLis本体中特有标注属性（目前发现是这样）
        self.lexical_annotation_uris.add("https://perkapp.fbk.eu/helis/ontology/core#commonSense")

    
    def get_prefered_label_annotation_iris(self):
        return self.main_label_uris
    

    def get_synonyms_annotation_iris(self):
        return self.synonym_label_uris
    

    def get_lexical_annotation_iris(self):
        return self.lexical_annotation_uris
        