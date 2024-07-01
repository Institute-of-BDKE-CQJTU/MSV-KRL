import os
import json
import torch
import rdflib
import pickle
import random
import argparse
import numpy as np
from lib.Helper import Helper
from lib.models.MTL import MTL
from lib.Subgraph import Subgraph
from ordered_set import OrderedSet
from collections import defaultdict
from lib.Projection import Projection
from rdflib.namespace import RDF, RDFS
from torch.utils.data import DataLoader
from lib.Random_Walk import Random_Walk
from lib.Bert_Dataset import Bert_Dataset
from lib.models.PretrainBertForMLM import PretrainBertForMLM
from lib.Label import URI_parse, pre_process_words, label_item
from lib.MTL_Datasets import MTL_Train_Dataset, MTL_Evaluate_Dataset, MTL_Valid_Dataset
from transformers import AutoTokenizer, BertConfig, DataCollatorForLanguageModeling, BertForMaskedLM



parser = argparse.ArgumentParser()
try:
    with open('./configs/helis_config.json', 'r') as json_file:
    # with open('./configs/go_config.json', 'r') as json_file:
        loaded_args = json.load(json_file)
        args = argparse.Namespace(**loaded_args)
except FileNotFoundError:
    args = parser.parse_args()


ontology_file_name_suffix = ".nt" if args.ontology_name in ["foodon", "go"] else ".xml"
device = "cuda" if torch.cuda.is_available() else "cpu"
ontology_name = args.ontology_name
ontology_file_path = args.ontology_file_path
related_file_save_path = args.related_file_save_path
lm_file_path = args.lm_file_path
step_option = args.step_option  ## 可以为multi_views、fine_tune、multi_task



projection = Projection(ontology_file_path=ontology_file_path)

########
## 1. Based on OWL2Vec* method to build corpus for BERT pre-training
if step_option == "multi_views":
    ## (1). Transform ontology file to RDF/XML file and N-Triple file(Convenient for debug)
    projection.extract_and_projection()
    projection.save_rdf_graph_to_file(rdf_file_dir=related_file_save_path, ontology_name=ontology_name)


    ## (2). Extract classes, individuals, object properties and data properties' uri
    projection.extract_uris()
    obj_properties = projection.get_obj_props()
    data_properties = projection.get_data_props()
    anno_properties = projection.get_anno_props()
    classes = projection.get_classes()
    individuals = projection.get_individuals()

    # Save entities and props as pickle
    with open(related_file_save_path+"entities_props_dict.pkl", 'wb') as f:
        entities_props_dict = {"classes": classes, "individuals": individuals, "obj_props": obj_properties, "data_props": data_properties}
        pickle.dump(entities_props_dict, f)

    # Only classes and individuals in owl2vec*, but properties have "label" annotation as well, so I added properties
    entities_and_props = classes.union(individuals).union(obj_properties).union(anno_properties).union(data_properties).union((str(RDF.type), str(RDFS.subClassOf)))
    # Save entities and props to file
    with open(related_file_save_path+"entities_props.txt", 'w') as f:
        for e in entities_and_props:
            f.write('%s\n' % e)


    ## (3). Extract axioms in Manchester syntax(Including subClassOf/equivalentClass restriction, object/data properties, subClassOf and type)
    projection.create_manchester_syntax_axiom()
    with open(related_file_save_path+"axioms.txt", 'w') as f:
        for ax in projection.axioms_manchester:
            f.write('%s\n' % ax)


    ## (4). Extract label, definition, comment and other annotation properties
    print("Start indexing annotation properties' value\n")
    uri_labels_dict = dict()
    annotations_list = list()   
    projection.get_annotations()
    print(f"entity num with label annotation: {len(projection.entity_to_labels_dict)}, entity num with all annotation properties: {len(projection.entity_to_all_annotations_dict)}")
    for eop in entities_and_props:
        if eop in projection.entity_to_labels_dict and len(projection.entity_to_labels_dict[eop]) > 0:
            labels_list = list(projection.entity_to_labels_dict[eop])[0]
            uri_labels_dict[eop] = pre_process_words(words=labels_list.split())

    for eop in entities_and_props:
        if eop in projection.entity_to_all_annotations_dict:
            for value in projection.entity_to_all_annotations_dict[eop]:
                if (value is not None) and (not (eop in projection.entity_to_labels_dict and value in projection.entity_to_labels_dict[eop])):
                    annotation = [eop] + value.split()  
                    annotations_list.append(annotation)

    # Save annotation properties value to file
    with open(related_file_save_path+"annotations.txt", 'w') as f:
        for e in projection.entity_to_labels_dict:
            for v in projection.entity_to_labels_dict[e]:
                f.write('%s label %s\n' % (e, v))
        for a in annotations_list:
            f.write('%s\n' % ' '.join(a))
    print("End indexing annotation properties' value\n")
    

    ## (5). Generate uri_sentences
    print("Start generating URI sentences")
    uri_sentences = list()
    rw_sentences = list()
    axiom_sentences = list()

    # Step 1: Get random walk URI sentences
    if ontology_name == "helis":
        rdf_file_path = related_file_save_path+ontology_name+'.xml'
    elif ontology_name in ["foodon", "go"]:
        rdf_file_path = related_file_save_path+ontology_name+'.nt'
    random_walk = Random_Walk(rdf_file_path=rdf_file_path, walk_depth=4, classes=classes.union(individuals))  # 這裡classes參數的目的是為了保證隨機遊走是從類和實例出發
    walks = random_walk.get_walks()
    print('Extracted %d walks for %d seed entities' % (len(walks), len(classes.union(individuals))))
    rw_sentences += [list(map(str, x)) for x in walks]

    # Step 2: Get built axioms from file
    if os.path.exists(related_file_save_path+'axioms.txt'):
        for line in open(related_file_save_path+'axioms.txt').readlines():
            axiom_sentence = [item for item in line.strip().split()]
            axiom_sentences.append(axiom_sentence)
    print('Extracted %d axiom sentences' % len(axiom_sentences))

    # Step 3: Combine the two parts
    uri_sentences = rw_sentences + axiom_sentences  

    
    ## (6). To generate sentences that includes all its annotations based on URI sentences and generate uri2anno dict
    print("Start generate all annotation sentences ...")
    entity2annotations_dict = projection.entity_to_all_annotations_dict
    entity_to_text_dict = dict()
    for sentence in uri_sentences:
        for i, uri in enumerate(sentence):
            anno_temp = ""
            try:
                if ontology_name == "helis":
                    annos = entity2annotations_dict[uri]
                    max_len = 0
                    for anno in annos:
                        if len(anno) > max_len:
                            anno_temp = anno
                            max_len = len(anno)
                elif ontology_name in ["foodon", "go"]:
                    annos = uri_labels_dict[uri]    
                    anno_temp = " ".join(annos)
            except:
                anno_temp = label_item(item=uri, uri_labels_dict=uri_labels_dict)
                anno_temp = " ".join(anno_temp)

            if ontology_name == "helis":
                if uri not in entity_to_text_dict and "#" in uri:
                    entity_to_text_dict[uri] = anno_temp 
            elif ontology_name in ["foodon", "go"]:
                ## foodon本体中实体uri没有‘#’，用这个来判断
                if uri not in entity_to_text_dict and "http://" in uri:
                    entity_to_text_dict[uri] = anno_temp 

    with open(related_file_save_path+'entity_to_text_dict.pkl', 'wb') as f:
        pickle.dump(entity_to_text_dict, f)
    print("End generate all annotation sentences\n")


    ## (7). Generate subgraphs and subgraphs' random walk, namely heirarchy structure information of specific relation like 'type' etc.
    # Step 1: Generate subgraphs
    print("Start generate RDF file's 5 subgraphs ...")
    subgraph = Subgraph(related_file_save_path + args.ontology_name + ontology_file_name_suffix)
    subgraph.generate_subgraphs(classes, individuals, obj_properties, data_properties, related_file_save_path+'subgraphs/')
    print("End generate subgraphs\n")
    
    # Step 2: Generate subClassOf subgraph's random walk
    # get random walk
    print("Start generate subClassOf url sentences")
    rw_sentences = list()
    rw = Random_Walk(rdf_file_path=related_file_save_path+'subgraphs/subclassof.xml', walk_depth=4, classes=classes)  # 這裡classes參數的目的是為了保證隨機遊走是從類和實例出發
    walks = rw.get_walks()
    print('Extracted %d walks for %d seed entities' % (len(walks), len(classes)))
    rw_sentences += [list(map(str, x)) for x in walks]
    # replace uri with its annotation
    subclassof_anno_sentences = list()
    for sentence in rw_sentences:
        if len(sentence) < 3:
            continue
        temp_list = [entity_to_text_dict[uri] for uri in sentence]
        subclassof_anno_sentences.append(temp_list)
    with open(related_file_save_path+'subgraphs/subclassof_anno_sentences.txt', 'w') as f:
        for sentence in subclassof_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n')
    print(f"End generate subClassOf url sentences, length: {len(subclassof_anno_sentences)}\n")

    # Step 3: Generate class property subgraph's random walk
    # get random walk
    print("Start generate class property url sentences")
    rw_sentences = list()
    rw = Random_Walk(rdf_file_path=related_file_save_path+'subgraphs/class_props.xml', walk_depth=4, classes=classes)  # 這裡classes參數的目的是為了保證隨機遊走是從類和實例出發
    walks = rw.get_walks()
    print('Extracted %d walks for %d seed entities' % (len(walks), len(classes)))
    rw_sentences += [list(map(str, x)) for x in walks]
    # replace uri with its annotation
    class_props_anno_sentences = list()
    for sentence in rw_sentences:
        if len(sentence) < 3:
            continue
        temp_list = [entity_to_text_dict[uri] for uri in sentence]
        class_props_anno_sentences.append(temp_list)
    with open(related_file_save_path+'subgraphs/class_props_anno_sentences.txt', 'w') as f:
        for sentence in class_props_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n')
    print(f"End generate class property url sentences, length: {len(class_props_anno_sentences)}\n")

    # Step 4: Generate individual type subgraph's random walk
    # get random walk
    print("Start generate individual type url sentences")
    rw_sentences = list()
    rw = Random_Walk(rdf_file_path=related_file_save_path+'subgraphs/ind_type.xml', walk_depth=4, classes=individuals)  # 這裡classes參數的目的是為了保證隨機遊走是從類和實例出發
    walks = rw.get_walks()
    print('Extracted %d walks for %d seed entities' % (len(walks), len(individuals)))
    rw_sentences += [list(map(str, x)) for x in walks]
    # replace uri with its annotation
    ind_type_anno_sentences = list()
    for sentence in rw_sentences:
        if len(sentence) < 3:
            continue
        temp_list = [entity_to_text_dict[uri] for uri in sentence]
        ind_type_anno_sentences.append(temp_list)
    with open(related_file_save_path+'subgraphs/ind_type_anno_sentences.txt', 'w') as f:
        for sentence in ind_type_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n')
    print(f"End generate individual type url sentences, length: {len(ind_type_anno_sentences)}\n")

    # Step 5: Generate individual object property subgraph's random walk
    # get random walk
    print("Start generate individual object property url sentences")
    rw_sentences = list()
    rw = Random_Walk(rdf_file_path=related_file_save_path+'subgraphs/obj_props.xml', walk_depth=4, classes=individuals)  # 這裡classes參數的目的是為了保證隨機遊走是從類和實例出發
    walks = rw.get_walks()
    print('Extracted %d walks for %d seed entities' % (len(walks), len(individuals)))
    rw_sentences += [list(map(str, x)) for x in walks]
    # replace uri with its annotation
    obj_props_anno_sentences = list()
    for sentence in rw_sentences:
        if len(sentence) < 3:
            continue
        temp_list = [entity_to_text_dict[uri] for uri in sentence]
        obj_props_anno_sentences.append(temp_list)
    with open(related_file_save_path+'subgraphs/obj_props_anno_sentences.txt', 'w') as f:
        for sentence in obj_props_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n')
    print(f"End generate individual object property url sentences, length: {len(obj_props_anno_sentences)}\n")

    # Step 6: Generate individual data property subgraph's random walk
    # get random walk
    print("Start generate individual data property url sentences")
    rw_sentences = list()
    rw = Random_Walk(rdf_file_path=related_file_save_path+'subgraphs/data_props.xml', walk_depth=4, classes=individuals)  # 這裡classes參數的目的是為了保證隨機遊走是從類和實例出發
    walks = rw.get_walks()
    print('Extracted %d walks for %d seed entities' % (len(walks), len(individuals)))
    rw_sentences += [list(map(str, x)) for x in walks]
    # replace uri with its annotation
    data_props_anno_sentences = list()
    for sentence in rw_sentences:
        if len(sentence) < 3:
            continue
        temp_list = list()
        for uri in sentence:
            try:
                anno_info = entity_to_text_dict[uri]
                temp_list.append(anno_info)
            except:
                temp_list.append(uri)
        data_props_anno_sentences.append(temp_list)
    with open(related_file_save_path+'subgraphs/data_props_anno_sentences.txt', 'w') as f:
        for sentence in data_props_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n')
    print(f"End generate individual data property url sentences, length: {len(data_props_anno_sentences)}\n")

    # Step 7: Translate axiom to annotation
    print("Start translate axiom sentences to annotation")
    axiom_anno_sentences = list()
    for sentence in axiom_sentences:
        if len(sentence) < 3:
            continue
        temp_list = list()
        for uri in sentence:
            try:
                anno_info = entity_to_text_dict[uri]
                temp_list.append(anno_info)
            except:
                temp_list.append(uri)
        axiom_anno_sentences.append(temp_list)
    with open(related_file_save_path+'subgraphs/axiom_anno_sentences_sep.txt', 'w') as f:
        for sentence in axiom_anno_sentences:
            f.write(" [SEP] ".join(sentence) + '\n') 
    print(f"End translate axiom sentences to annotation, length: {len(axiom_anno_sentences)}\n")

    # Step 8: Union all annotation sentences 
    all_anno_sentences = subclassof_anno_sentences + class_props_anno_sentences + ind_type_anno_sentences + obj_props_anno_sentences + data_props_anno_sentences
    print(f"Total annotation sentence num: {len(all_anno_sentences)}")
    all_anno_sentences = set([" [SEP] ".join(sentence) for sentence in all_anno_sentences])
    print(f"Repetitive sentences are removed, total num: {len(all_anno_sentences)}")
    with open(related_file_save_path+'subgraphs/all_anno_sentences_sep.txt', 'w') as f:
        for sentence in all_anno_sentences:
            f.write(sentence + '\n')
    print(f"End generate all subgraphs' annotation sentences (repetitive sentences are removed), total num: {len(all_anno_sentences)}\n")
    


    
########
## 2. Based on the five types sentences to pretrain bert
if step_option == "fine_tune":
    ## (1). Load bert pre-training data and new vocabs(classes, individuals, object properties and data properties, also include type and subClassOf) 
    print("Load original pre-training data file ...")
    training_source_data = list()
    with open(related_file_save_path+'subgraphs/all_anno_sentences_sep.txt') as f:
        for sentence in f.readlines():
            training_source_data.append(sentence.replace('\n', ''))
    with open(related_file_save_path+'subgraphs/axiom_anno_sentences_sep.txt') as f:
        for sentence in f.readlines():
            training_source_data.append(sentence.replace('\n', ''))
    inputs_dataset = training_source_data
    print(f"Total samples: {len(inputs_dataset)}\n")
    

    ## (2). Init tokenizer
    print("Initialize a new tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(lm_file_path)
    print(f"Tokenizer vocab size: {len(tokenizer)}\n")


    ## (3). Init DataLoader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.lm_fine_tune_mlm_probability)
    bert_dataset = Bert_Dataset(inputs_dataset, tokenizer, data_collator)
    bert_dataloader = DataLoader(bert_dataset, batch_size=args.fine_tune_batch_size, shuffle=True, num_workers=args.lm_fine_tune_num_workers, pin_memory=True)


    ## (4). Init bert model
    bert_config = BertConfig.from_pretrained(lm_file_path)
    model = PretrainBertForMLM(bert_config, lm_file_path, device)
    # print(model)
    model.to(device)


    ## (5). Start pre-training
    args_1 = dict()
    args_1['lr'] = args.fine_tune_lr
    args_1['epochs'] = args.fine_tune_epochs
    args_1['batch_size'] = args.fine_tune_batch_size
    args_1['model_save_base_path'] = related_file_save_path + args.lm_save_path
    args_1['train_records_save_base_path'] = related_file_save_path + args.lm_fine_tune_records_save_path
    model.pre_training(bert_dataloader, args_1)




########
## 3. Train MTL 
if step_option == "multi_task":
    ## (1). Init Embeddings
    if args.is_extract_knowledge:
        Helper.knowledge_extraction(args, device)


    ## (2). Get nodes and edges from file
    print("Get rdf file nodes and edges ...")
    nodes = OrderedSet()
    edges = OrderedSet()
    with open(related_file_save_path+'nodes.txt', 'r') as f:
        for node in f.readlines():
            nodes.add(node.replace('\n', ''))
    with open(related_file_save_path+'edges.txt', 'r') as f:
        for edge in f.readlines():
            edges.add(edge.replace('\n', ''))
    print(f"End getting nodes and edges, nodes num: {len(nodes)}, edges num: {len(edges)}")


    ## (3). Get BERT lexical info embeddings to init entity and relation embeddings
    # Step 1: Get embeddings
    print("Initial entity and property embeddings by pre-trained BERT ...")
    with open(related_file_save_path + args.embeddings_save_file_name, 'rb') as f:
        embedding_dict = pickle.load(f)
    
    # Step 2: Generate nodes and edges init embedding
    entity_embeddings = list()
    relation_embeddings = list()
    for node in nodes:
        embedding = embedding_dict[node]
        entity_embeddings.append(embedding)
    for edge in edges:
        embedding = embedding_dict[edge]
        relation_embeddings.append(embedding)
    print(f"Got init embeddings total num: {len(embedding_dict)}, nodes embedding num: {len(entity_embeddings)}, edges embedding num: {len(relation_embeddings)}")


    ## (4). Process origin triple, generate training data and initialize data
    # Step 1: define hyper parameters
    args.num_ent = len(nodes)
    args.num_rel = len(edges)
    # print(args)

    # Step 2: build index dict
    entity2idx = {e: idx for idx, e in enumerate(nodes)}
    relation2idx = {r: idx for idx, r in enumerate(edges)}
    # Inverse relation
    relation2idx.update({rel + '_reverse': idx + len(relation2idx) for idx, rel in enumerate(nodes)})
    # Inverse projection
    idx2entity = {value: key for key, value in entity2idx.items()}
    idx2relation = {value: key for key, value in relation2idx.items()}
    # print(len(entity2idx),len(idx2entity),len(relation2idx),len(idx2relation))

    # Step 4: Generate MTL model
    model = None
    if args.evaluate_exist_model:
        model = torch.load(args.related_file_save_path + args.mtl_model_save_path + args.evaluate_exist_model_name)
    else:
        model = MTL(entity_embeddings, relation_embeddings, idx2entity, idx2relation, args, device)
    model.to(device)


    ## (5). Train MTL model and get their embeddings
    # Step 1: Load train, validate and test data
    train_triples = []
    valid_triples = []
    test_triples = []
    with open(related_file_save_path+'triple_train_with_owl2vec.txt') as f:  #mtl_datasets/
        for line in f.readlines():
            triple_list = line.replace('\n', '').split('\t')
            train_triples.append(triple_list)
    with open(related_file_save_path+'triple_valid_with_owl2vec.txt') as f:
        for line in f.readlines():
            triple_list = line.replace('\n', '').split('\t')
            valid_triples.append(triple_list)
    with open(related_file_save_path+'triple_test_with_owl2vec.txt') as f:
        for line in f.readlines():
            triple_list = line.replace('\n', '').split('\t')
            if args.test_all:
                test_triples.append(triple_list)  
            else:
                if triple_list[1] == str(RDF.type):
                    test_triples.append(triple_list)

    # Step 2: Init Dataset and DataLoader
    # 初始化Helper中的类对象，比如classes
    Helper.get_entities_and_props(related_file_save_path+'entities_props_dict.pkl', related_file_save_path + args.ontology_name + ontology_file_name_suffix)
    train_dataset = MTL_Train_Dataset(train_triples, entity2idx, relation2idx)
    valid_dataset = MTL_Valid_Dataset(valid_triples, entity2idx, relation2idx)
    test_dataset = MTL_Evaluate_Dataset(test_triples, entity2idx, relation2idx, args)
    train_dataloader = DataLoader(train_dataset, args.batch, shuffle=True, num_workers=args.mtl_num_workers, collate_fn=Helper.custom_train_dataloader_collate_func, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch*2, shuffle=True, num_workers=args.mtl_num_workers, collate_fn=Helper.custom_valid_dataloader_collate_func, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=True, num_workers=args.mtl_num_workers, collate_fn=Helper.custom_evaluate_dataloader_collate_func, pin_memory=True)


    if not args.evaluate_exist_model:
        epoch_losses = model.run_training(train_dataloader, valid_dataloader, test_dataloader)


    # 模型验证
    model.evaluate(test_dataloader)