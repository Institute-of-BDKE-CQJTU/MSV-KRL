import sys
import torch
import pickle
import rdflib
import random
import numpy as np
from collections import defaultdict
from rdflib.namespace import RDF, RDFS
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForMaskedLM, BertConfig


class Helper(object):
    ## 帮助类，用于实现多处地方都需要的通用工具函数
    # 三元组类型静态变量
    CLASS_SUBSUMPTION = 1
    CLASS_MEMBERSHIP = 2
    CLASS_PROPERTY = 3
    INDIVIDUAL_OBJECT_PROPERTY = 4
    INDIVIDUAL_DATA_PROPERTY = 5
    task_name_dict = {
        1: "Class Subsumption",
        2: "Class Membership",
        3: "Class Property",
        4: "Individual Object Property",
        5: "Individual Data Property"
    }

    # 实体和属性
    classes = list()
    individuals = list()
    obj_props = list()
    data_props = list()
    all_triples = list()


    def __init__(self) -> None:
        pass


    @classmethod
    def get_entities_and_props(self, pickle_file_path, rdf_file_path):
        with open(pickle_file_path, 'rb') as f:
            entity_props_dict = pickle.load(f)

        Helper.classes = list(entity_props_dict['classes'])
        Helper.individuals = list(entity_props_dict['individuals'])
        Helper.obj_props = list(entity_props_dict['obj_props'])
        Helper.data_props = list(entity_props_dict['data_props'])
        graph = rdflib.Graph().parse(rdf_file_path)
        for s, p, o in graph:
            triple_list = [str(s), str(p), str(o)]
            Helper.all_triples.append(",".join(triple_list))
                
        
    
    @classmethod
    def judge_triple_type(self, triple):
        if len(Helper.classes) == 0:
            print("实体和属性为空，请先调用get_entities_and_props()函数获取实体和属性！")
            sys.exit()

        s, p, o = triple[0], triple[1], triple[2]
        if p == str(RDFS.subClassOf):
            return Helper.CLASS_SUBSUMPTION
        elif p == str(RDF.type):
            return Helper.CLASS_MEMBERSHIP
        elif (p in Helper.obj_props or p in Helper.data_props) and s in Helper.classes:
            return Helper.CLASS_PROPERTY
        elif p in Helper.obj_props and s in Helper.individuals and o in Helper.individuals:
            return Helper.INDIVIDUAL_OBJECT_PROPERTY
        elif p in Helper.data_props and s in Helper.individuals:
            return Helper.INDIVIDUAL_DATA_PROPERTY
        else:
            return 0


    @classmethod
    def get_negative_sample(self, triple):
        ## @classmethod注解和@staticmethod注解的区别就是的第一个参数通常定义为“self”，和类中没有注解的函数的区别就是
        ## 该函数可以直接通过类名进行访问
        triple_type = Helper.judge_triple_type(triple=triple)
        
        replace_index = -1 # 负样本待更新位置下标
        neg_sample_candidate_entities = list()  # 负样本候选实体只能是类或实例
        if triple_type == Helper.CLASS_SUBSUMPTION or triple_type == Helper.CLASS_MEMBERSHIP or triple_type == Helper.CLASS_PROPERTY:
            if triple[2] in Helper.classes:
                replace_index = 2
            else:
                replace_index = 0
            neg_sample_candidate_entities = Helper.classes
        elif triple_type == Helper.INDIVIDUAL_OBJECT_PROPERTY or triple_type == Helper.INDIVIDUAL_DATA_PROPERTY:
            replace_index = 0
            neg_sample_candidate_entities = Helper.individuals

        is_still_pos = True
        random_entity = triple[replace_index]
        if len(neg_sample_candidate_entities) == 0:
            print(triple)
        # print(neg_sample_candidate_entities[:5])
        while(is_still_pos == True or random_entity == triple[replace_index]):
            random_index = random.randint(0, len(neg_sample_candidate_entities) - 1)
            random_entity = neg_sample_candidate_entities[random_index]
            neg_sample = triple[:]
            neg_sample[replace_index] = random_entity
            is_still_pos = Helper.judge_pos_or_neg(neg_sample)

        return neg_sample


    @classmethod
    def judge_pos_or_neg(self, sample):
        if len(Helper.all_triples) == 0:
            print("请先调用get_entities_and_props()函数获取所有的三元组")
            sys.exit()

        is_still_pos = False
        if ",".join(sample) in Helper.all_triples:
            is_still_pos = True

        return is_still_pos
    

    @classmethod
    def custom_train_dataloader_collate_func(self, batch):
        samples = list()
        labels = list()
        # 单个sample包含了正例和负例，第一个元素为正例，第二个元素为它的负例，每一个正例或负例元素包含了样本和标签
        for sample in batch:
            samples.append(sample[0][0])    # 正例
            labels.append(sample[0][1])     # 正例标签

            samples.append(sample[1][0])    # 负例
            labels.append(sample[1][1])     # 负例标签


        return np.array(samples), np.array(labels, dtype=np.float32) 
    

    @classmethod
    def get_triples_with_candidate_entity(self, triple):
        triple_type = Helper.judge_triple_type(triple=triple)
        
        replace_index = -1 # 负样本待更新位置下标
        candidate_entities = list()  # 负样本候选实体只能是类或实例
        if triple_type == Helper.CLASS_SUBSUMPTION or triple_type == Helper.CLASS_MEMBERSHIP or triple_type == Helper.CLASS_PROPERTY:
            if triple[2] in Helper.classes:
                replace_index = 2
            else:
                replace_index = 0
            candidate_entities = Helper.classes
        elif triple_type == Helper.INDIVIDUAL_OBJECT_PROPERTY or triple_type == Helper.INDIVIDUAL_DATA_PROPERTY:
            replace_index = 0
            candidate_entities = Helper.individuals

        true_candidate_entity = list()
        candidate_triples = list()
        true_candidate_entity.append(triple[-1])
        candidate_triples.append(triple)    # 首先需要添加正例到候选实体三元组中去，最为其中的一员
        for entity in candidate_entities:
            if entity == triple[replace_index]:
                # 过滤正确实体
                continue

            candidate_triple = triple[:]
            candidate_triple[replace_index] = entity

            # 采用和transE，owl2vec*中在进行模型验证与测试一样的过滤设定（filter setting），即候选实体三元组不能是在训练集、测试集以及验证集中已有的三元组
            if not Helper.judge_pos_or_neg(candidate_triple):
                true_candidate_entity.append(entity)
                candidate_triples.append(candidate_triple)

        return true_candidate_entity, candidate_triples
    

    @classmethod
    def custom_valid_dataloader_collate_func(self, batch):
        samples = list()
        labels = list()
        for item in batch:
            samples.append(item[0])
            labels.append(item[1])

        return np.array(samples), np.array(labels, dtype=np.float32)

    @classmethod
    def custom_evaluate_dataloader_collate_func(self, batch):
        
        return batch
    

    @classmethod
    def knowledge_extraction(self, args, device):
        is_mean_pooling = args.is_mean_pooling
        related_file_save_path = args.related_file_save_path

        ## 经过训练的BERT
        bert_config = BertConfig.from_pretrained(related_file_save_path + args.lm_save_path + args.fine_tuned_lm)
        bert_config.output_hidden_states = is_mean_pooling
        tokenizer = BertTokenizer.from_pretrained(args.lm_file_path)
        bert_model = BertForMaskedLM.from_pretrained(related_file_save_path + args.lm_save_path + args.fine_tuned_lm, config=bert_config).bert
        bert_model.to(device)
        for param in bert_model.parameters():
            param.requires_grad = False

        if args.ontology_name == "helis":
            ontology_suffix = ".xml"
        elif args.ontology_name in ["foodon", "go"]:
            ontology_suffix = ".nt"
        graph = rdflib.Graph()
        graph.parse(related_file_save_path + args.ontology_name + ontology_suffix)

        # 包含了所有实体的uri，除了标注属性 [目前没有什么用]
        entity_prop_uncased = []
        entity_prop_cased = []
        with open(related_file_save_path+'entities_props.txt', 'r') as f:
            for word in f.readlines():
                word = word.replace('\n', '')
                entity_prop_cased.append(word)
                entity_prop_uncased.append(word.lower())

        # 理论上包含所有出现过在本体中的实体uri对应的标注属性值（对于有多个标注属性的取包含信息最多的那个，没有的取其uri中“#”后面的）
        with open(related_file_save_path+'entity_to_text_dict.pkl', 'rb') as f:
            entity_prop_2_anno_dict = pickle.load(f)

        print("Initial entity and property embeddings by fine tuned BERT ...")
        nodes = set()
        edges = set()
        embedding_dict = defaultdict(list)
        count = 0
        print("\tGet all embedding from bert ...")
        # 这里通过循环整个rdf中的三元组获取实体和属性而不是循环entities_props.txt的原因是，三元组中是包含了数据属性的，对于字面量和数值类数据这当中没有
        for s, p, o in graph:
            ## 跳过空白节点
            if isinstance(s, rdflib.term.BNode) or isinstance(p, rdflib.term.BNode) or isinstance(o, rdflib.term.BNode):
                continue
            if str(s).strip().replace('\n', '') == "" or str(p).strip().replace('\n', '') == "" or str(o).strip().replace('\n', '') == "":
                continue

            triple_list = [str(s).strip().replace('\n', ''), str(p).strip().replace('\n', ''), str(o).strip().replace('\n', '')]
            count += 1
            nodes.add(str(s).strip().replace('\n', ''))
            nodes.add(str(o).strip().replace('\n', ''))
            edges.add(str(p).strip().replace('\n', ''))

            sentences = []
            for item in triple_list:
                try:
                    sentences.append(entity_prop_2_anno_dict[item])
                except:
                    sentences.append(item)
            print(f"Processing No.{count} triple: {sentences} ...")

            sample = " [SEP] ".join(sentences)
            inputs_ids = tokenizer.encode_plus(sample, return_tensors="pt", max_length=512)
            batch_encoding = inputs_ids['input_ids'].to(device)
            outputs = bert_model(input_ids=batch_encoding)
            
            # 创建一个布尔张量，表示张量中是否与目标数字相等
            bool_tensor = torch.eq(batch_encoding[0], 102)
            # 使用 torch.nonzero() 找到非零元素的下标
            indices = torch.nonzero(bool_tensor).reshape((1, -1))[0].tolist()

            ## 获取实体向量
            hidden_states = outputs.hidden_states[1:]   # 没有包括初始Embedding层
            temp_s_e_list = list()
            temp_p_e_list = list()
            temp_o_e_list = list()
            for hidden_state in hidden_states:
                hidden_state = hidden_state[0]
                s_tokens_e, p_tokens_e, o_tokens_e = hidden_state[1:indices[0]], hidden_state[indices[0]+1:indices[1]], hidden_state[indices[1]+1:indices[2]]
                for s_token_e in s_tokens_e:
                    temp_s_e_list.append(s_token_e)
                for p_token_e in p_tokens_e:
                    temp_p_e_list.append(p_token_e)
                for o_token_e in o_tokens_e:
                    temp_o_e_list.append(o_token_e)
            
            s_embedding = torch.mean(torch.stack(temp_s_e_list), dim=0)
            p_embedding = torch.mean(torch.stack(temp_p_e_list), dim=0)
            o_embedding = torch.mean(torch.stack(temp_o_e_list), dim=0)

            embedding_dict[triple_list[0]].append(s_embedding)
            embedding_dict[triple_list[1]].append(p_embedding)
            embedding_dict[triple_list[2]].append(o_embedding)

        final_embedding_dict = dict()
        for key in embedding_dict:
            embedding = torch.mean(torch.stack(embedding_dict[key]), dim=0).cpu().numpy()
            final_embedding_dict[key] = embedding

        print("\tSave embedding dict object to file by pickle ...")
        with open(related_file_save_path + args.embeddings_save_file_name, 'wb') as f:
            pickle.dump(final_embedding_dict, f)
        print(f"End init entity and property embedding by pre-trained BERT, total entity and prop: {len(final_embedding_dict)}")

        with open(related_file_save_path+'nodes.txt', 'w') as f:
            for node in nodes:
                f.write(node+'\n')
        with open(related_file_save_path+'edges.txt', 'w') as f:
            for edge in edges:
                f.write(edge+'\n')


    @classmethod
    def build_mtl_training_datasets(self, args):
        related_file_save_path = args.related_file_save_path

        train_ratio = 0.7
        valid_ratio = 0.1
        test_ratio = 0.2

        print("开始处理 ...")
        if args.ontology_name not in ["foodon", "go"]:
            subclassof_triple_list = list()
            subclassof_graph = rdflib.Graph().parse(related_file_save_path+'subgraphs/subclassof.xml')
            for s, p, o in subclassof_graph:
                subclassof_triple_list.append([str(s), str(p), str(o)])

            # (2) 对subClassOf三元组划分数据集
            subclassof_train = list()
            subclassof_valid = list()
            subclassof_test = list()
            try:
                subclassof_train, temp_data = train_test_split(subclassof_triple_list, test_size=(1-train_ratio))
                subclassof_valid, subclassof_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))

                print(f"subClassOf处理完毕，总共三元组：{len(subclassof_triple_list)}，训练集：{len(subclassof_train)}，验证集：{len(subclassof_valid)}，测试集：{len(subclassof_test)}")
                print(f"比例为：训练集:验证集:测试集 = {len(subclassof_train)/len(subclassof_triple_list):.4f}:{len(subclassof_valid)/len(subclassof_triple_list):.4f}:{len(subclassof_test)/len(subclassof_triple_list):.4f}\n")
            except:
                print("当前本体无此关系三元组\n")
        else:
            subclassof_train = list()
            subclassof_valid = list()
            subclassof_test = list()
            with open(args.owl2vec_data_file_path + args.ontology_name + "/train.csv", 'r') as f:
                for line in f.readlines():
                    triple_temp = line.strip().split(',')
                    if triple_temp[2] == "1":
                        # 只添加正例，我的方法中负例是动态生成的
                        subclassof_train.append([triple_temp[0], str(RDFS.subClassOf), triple_temp[1]])
            with open(args.owl2vec_data_file_path + args.ontology_name + "/valid.csv", 'r') as f:
                for line in f.readlines():
                    triple_temp = line.strip().split(',')
                    subclassof_valid.append([triple_temp[0], str(RDFS.subClassOf), triple_temp[1]])
            with open(args.owl2vec_data_file_path + args.ontology_name + "/test.csv", 'r') as f:
                for line in f.readlines():
                    triple_temp = line.strip().split(',')
                    subclassof_test.append([triple_temp[0], str(RDFS.subClassOf), triple_temp[1]])
            owl2vec_total_count = len(subclassof_train) + len(subclassof_valid) + len(subclassof_test)

            print(f"subClassOf处理完毕，总共三元组：{owl2vec_total_count}，训练集：{len(subclassof_train)}，验证集：{len(subclassof_valid)}，测试集：{len(subclassof_test)}")
            print(f"比例为：训练集:验证集:测试集 = {len(subclassof_train)/owl2vec_total_count:.4f}:{len(subclassof_valid)/owl2vec_total_count:.4f}:{len(subclassof_test)/owl2vec_total_count:.4f}\n")


        ####### 2、处理class props三元组
        # (1) 获取三元组列表
        class_prop_triple_list = list()
        class_prop_graph = rdflib.Graph().parse(related_file_save_path+'subgraphs/class_props.xml')
        for s, p, o in class_prop_graph:
            class_prop_triple_list.append([str(s), str(p), str(o)])

        # (2) 对class props三元组划分数据集
        # class_prop_train, temp_data = train_test_split(class_prop_triple_list, test_size=(1-train_ratio))
        # class_prop_valid, class_prop_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))

        # print(f"class props处理完毕，总共三元组：{len(class_prop_triple_list)}，训练集：{len(class_prop_train)}，验证集：{len(class_prop_valid)}，测试集：{len(class_prop_test)}")
        # print(f"比例为：训练集:验证集:测试集 = {len(class_prop_train)/len(class_prop_triple_list):.4f}:{len(class_prop_valid)/len(class_prop_triple_list):.4f}:{len(class_prop_test)/len(class_prop_triple_list):.4f}\n")
        # (2) 对class props三元组划分数据集
        class_prop_train = list()
        class_prop_valid = list()
        class_prop_test = list()
        try:
            class_prop_train, temp_data = train_test_split(class_prop_triple_list, test_size=(1-train_ratio))
            class_prop_valid, class_prop_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))

            print(f"class props处理完毕，总共三元组：{len(class_prop_triple_list)}，训练集：{len(class_prop_train)}，验证集：{len(class_prop_valid)}，测试集：{len(class_prop_test)}")
            print(f"比例为：训练集:验证集:测试集 = {len(class_prop_train)/len(class_prop_triple_list):.4f}:{len(class_prop_valid)/len(class_prop_triple_list):.4f}:{len(class_prop_test)/len(class_prop_triple_list):.4f}\n")
        except:
            print("当前本体无此关系三元组")


        ####### 3、处理indiviudal type三元组
        # (1) 获取三元组列表
        ## 当条件为True时划分自己的type三元组，否则加载owl2vec的type三元组
        if args.ontology_name != "helis":
            ind_type_triple_list = list()
            ind_type_graph = rdflib.Graph().parse(related_file_save_path+'subgraphs/ind_type.xml')
            for s, p, o in ind_type_graph:
                ind_type_triple_list.append([str(s), str(p), str(o)])

            # (2) 对indiviudal type三元组划分数据集
            # ind_type_train, temp_data = train_test_split(ind_type_triple_list, test_size=(1-train_ratio))
            # ind_type_valid, ind_type_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))

            # print(f"indiviudal type处理完毕，总共三元组：{len(ind_type_triple_list)}，训练集：{len(ind_type_train)}，验证集：{len(ind_type_valid)}，测试集：{len(ind_type_test)}")
            # print(f"比例为：训练集:验证集:测试集 = {len(ind_type_train)/len(ind_type_triple_list):.4f}:{len(ind_type_valid)/len(ind_type_triple_list):.4f}:{len(ind_type_test)/len(ind_type_triple_list):.4f}\n")
            ind_type_train = list()
            ind_type_valid = list()
            ind_type_test = list()
            try:
                ind_type_train, temp_data = train_test_split(ind_type_triple_list, test_size=(1-train_ratio))
                ind_type_valid, ind_type_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))

                print(f"indiviudal type处理完毕，总共三元组：{len(ind_type_triple_list)}，训练集：{len(ind_type_train)}，验证集：{len(ind_type_valid)}，测试集：{len(ind_type_test)}")
                print(f"比例为：训练集:验证集:测试集 = {len(ind_type_train)/len(ind_type_triple_list):.4f}:{len(ind_type_valid)/len(ind_type_triple_list):.4f}:{len(ind_type_test)/len(ind_type_triple_list):.4f}\n")
            except:
                print("当前本体无此关系三元组\n")
        else:
            ind_type_train = list()
            ind_type_valid = list()
            ind_type_test = list()
            with open(args.owl2vec_data_file_path + "helis/train.csv", 'r') as f:
                for line in f.readlines():
                    triple_temp = line.strip().split(',')
                    if triple_temp[2] == "1":
                        # 只添加正例，我的方法中负例是动态生成的
                        ind_type_train.append([triple_temp[0], str(RDF.type), triple_temp[1]])
            with open(args.owl2vec_data_file_path + "helis/valid.csv", 'r') as f:
                for line in f.readlines():
                    triple_temp = line.strip().split(',')
                    ind_type_valid.append([triple_temp[0], str(RDF.type), triple_temp[1]])
            with open(args.owl2vec_data_file_path + "helis/test.csv", 'r') as f:
                for line in f.readlines():
                    triple_temp = line.strip().split(',')
                    ind_type_test.append([triple_temp[0], str(RDF.type), triple_temp[1]])
            owl2vec_total_count = len(ind_type_train) + len(ind_type_valid) + len(ind_type_test)

            print(f"indiviudal type处理完毕，总共三元组：{owl2vec_total_count}，训练集：{len(ind_type_train)}，验证集：{len(ind_type_valid)}，测试集：{len(ind_type_test)}")
            print(f"比例为：训练集:验证集:测试集 = {len(ind_type_train)/owl2vec_total_count:.4f}:{len(ind_type_valid)/owl2vec_total_count:.4f}:{len(ind_type_test)/owl2vec_total_count:.4f}\n")


        ####### 4、处理obj props三元组
        # (1) 获取三元组列表
        obj_prop_triple_list = list()
        obj_prop_graph = rdflib.Graph().parse(related_file_save_path+'subgraphs/obj_props.xml')
        for s, p, o in obj_prop_graph:
            obj_prop_triple_list.append([str(s), str(p), str(o)])

        # (2) 对obj props三元组划分数据集
        # obj_prop_train, temp_data = train_test_split(obj_prop_triple_list, test_size=(1-train_ratio))
        # obj_prop_valid, obj_prop_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))

        # print(f"对obj props处理完毕，总共三元组：{len(obj_prop_triple_list)}，训练集：{len(obj_prop_train)}，验证集：{len(obj_prop_valid)}，测试集：{len(obj_prop_test)}")
        # print(f"比例为：训练集:验证集:测试集 = {len(obj_prop_train)/len(obj_prop_triple_list):.4f}:{len(obj_prop_valid)/len(obj_prop_triple_list):.4f}:{len(obj_prop_test)/len(obj_prop_triple_list):.4f}\n")
        obj_prop_train = list()
        obj_prop_valid = list()
        obj_prop_test = list()
        try:
            obj_prop_train, temp_data = train_test_split(obj_prop_triple_list, test_size=(1-train_ratio))
            obj_prop_valid, obj_prop_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))

            print(f"对obj props处理完毕，总共三元组：{len(obj_prop_triple_list)}，训练集：{len(obj_prop_train)}，验证集：{len(obj_prop_valid)}，测试集：{len(obj_prop_test)}")
            print(f"比例为：训练集:验证集:测试集 = {len(obj_prop_train)/len(obj_prop_triple_list):.4f}:{len(obj_prop_valid)/len(obj_prop_triple_list):.4f}:{len(obj_prop_test)/len(obj_prop_triple_list):.4f}\n")
        except:
            print("当前本体无此关系三元组\n")


        ####### 5、处理data props三元组
        # (1) 获取三元组列表
        data_prop_triple_list = list()
        data_prop_graph = rdflib.Graph().parse(related_file_save_path+'subgraphs/data_props.xml')
        for s, p, o in data_prop_graph:
            data_prop_triple_list.append([str(s), str(p), str(o)])

        # (2) 对data props三元组划分数据集
        # data_prop_train, temp_data = train_test_split(data_prop_triple_list, test_size=(1-train_ratio))
        # data_prop_valid, data_prop_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))

        # print(f"对data props处理完毕，总共三元组：{len(data_prop_triple_list)}，训练集：{len(data_prop_train)}，验证集：{len(data_prop_valid)}，测试集：{len(data_prop_test)}")
        # print(f"比例为：训练集:验证集:测试集 = {len(data_prop_train)/len(data_prop_triple_list):.4f}:{len(data_prop_valid)/len(data_prop_triple_list):.4f}:{len(data_prop_test)/len(data_prop_triple_list):.4f}\n")
        data_prop_train = list()
        data_prop_valid = list()
        data_prop_test = list()
        try:
            data_prop_train, temp_data = train_test_split(data_prop_triple_list, test_size=(1-train_ratio))
            data_prop_valid, data_prop_test = train_test_split(temp_data, test_size=test_ratio/(test_ratio+valid_ratio))

            print(f"对data props处理完毕，总共三元组：{len(data_prop_triple_list)}，训练集：{len(data_prop_train)}，验证集：{len(data_prop_valid)}，测试集：{len(data_prop_test)}")
            print(f"比例为：训练集:验证集:测试集 = {len(data_prop_train)/len(data_prop_triple_list):.4f}:{len(data_prop_valid)/len(data_prop_triple_list):.4f}:{len(data_prop_test)/len(data_prop_triple_list):.4f}\n")
        except:
            print("当前本体无此关系三元组\n")


        ###### 6、合并训练集、验证集、测试集
        train_data = subclassof_train + class_prop_train + ind_type_train + obj_prop_train + data_prop_train
        valid_data = subclassof_valid + class_prop_valid + ind_type_valid + obj_prop_valid + data_prop_valid
        test_data = subclassof_test + class_prop_test + ind_type_test + obj_prop_test + data_prop_test
        total_triple_num = len(train_data)+len(valid_data)+len(test_data)
        print(f"所有三元组理完毕，总共三元组：{total_triple_num}，训练集：{len(train_data)}，验证集：{len(valid_data)}，测试集：{len(test_data)}")
        print(f"比例为：训练集:验证集:测试集 = {len(train_data)/total_triple_num:.4f}:{len(valid_data)/total_triple_num:.4f}:{len(test_data)/total_triple_num:.4f}\n")


        ###### 7、保存训练集、验证集、测试集到文件
        with open(related_file_save_path+'triple_train_with_owl2vec.txt', 'w') as f:
            for triple_list in train_data:
                f.write("\t".join(triple_list) + '\n')
                
        with open(related_file_save_path+'triple_valid_with_owl2vec.txt', 'w') as f:
            for triple_list in valid_data:
                f.write("\t".join(triple_list) + '\n')

        with open(related_file_save_path+'triple_test_with_owl2vec.txt', 'w') as f:
            for triple_list in test_data:
                f.write("\t".join(triple_list) + '\n')
