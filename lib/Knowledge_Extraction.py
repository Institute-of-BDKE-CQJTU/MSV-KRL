import torch
import pickle
import rdflib
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from rdflib.namespace import RDF, RDFS
from transformers import BertTokenizer, BertForMaskedLM, BertConfig






is_mean_pooling = True
related_file_save_path = "/home/nlp/NLP-Group/XJC/code/OWL_Embedding_HeLis/result/HeLis/"


## 经过训练的BERT
bert_config = BertConfig.from_pretrained("/home/nlp/NLP-Group/XJC/code/OWL_Embedding_HeLis/result/HeLis/models/bert/bert_mlm_811_best")
bert_config.output_hidden_states = is_mean_pooling # 是否输出BERT base中所有12层中的最后一层输出
tokenizer = BertTokenizer.from_pretrained("/home/nlp/NLP-Group/XJC/code/OWL_Embedding_HeLis/data/bert_base")
bert_model = BertForMaskedLM.from_pretrained("/home/nlp/NLP-Group/XJC/code/OWL_Embedding_HeLis/result/HeLis/models/bert/bert_mlm_811_best", config=bert_config).bert

'''
## 原始BERT
bert_config = BertConfig.from_pretrained("/home/nlp/NLP-Group/XJC/code/OWL_Embedding_HeLis/data/bert_base")
bert_config.output_hidden_states = is_mean_pooling # 是否输出BERT base中所有12层中的最后一层输出
tokenizer = BertTokenizer.from_pretrained("/home/nlp/NLP-Group/XJC/code/OWL_Embedding_HeLis/data/bert_base")
bert_model = BertForMaskedLM.from_pretrained("/home/nlp/NLP-Group/XJC/code/OWL_Embedding_HeLis/data/bert_base", config=bert_config).bert
'''


# print(bert_model)
bert_model.to('cuda:0')
for param in bert_model.parameters():
    param.requires_grad = False

graph = rdflib.Graph()
graph.parse(related_file_save_path+'helis.xml')

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

print("Initial entity and property embeddings by pre-trained BERT ...")
nodes = set()
edges = set()
embedding_dict = defaultdict(list)
count = 0
print("\tGet all embedding from bert ...")
# 这里通过循环整个rdf中的三元组获取实体和属性而不是循环entities_props.txt的原因是，三元组中是包含了数据属性的，对于字面量和数值类数据这当中没有
for s, p, o in graph:
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
    inputs_ids = tokenizer.encode_plus(sample, return_tensors="pt")
    batch_encoding = inputs_ids['input_ids'].to('cuda:0')
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


    '''
    for i, sentence in enumerate(sentences):
        if triple_list[i] not in node_edge_embed_dict:
            inputs = tokenizer.encode_plus(sentence, return_tensors="pt")
            batch_encoding = inputs['input_ids'].to('cuda')
            outputs = bert_model(input_ids=batch_encoding)
            if is_mean_pooling:
                # hidden_states = outputs.hidden_states   # 包括了初始Embedding层
                hidden_states = outputs.hidden_states[1:]   # 没有包括初始Embedding层
                ## 我想这里的操作出错了，不应该是先池化单层所有token，再池化所有层，而是所有层中的所有token一起池化
                # # 每一层所有的token平均池化
                # hidden_states_token_mean_pool = torch.stack([torch.mean(hidden_state[0], dim=0) for hidden_state in hidden_states])
                # # 总共12层，进行平均池化
                # embedding = torch.mean(hidden_states_token_mean_pool, dim=0).cpu().numpy()
                tokens_from_all_layers = list()
                for hidden_state in hidden_states:
                    for j in range(hidden_state.shape[1]):
                        tokens_from_all_layers.append(hidden_state[0][j])
                embedding 
                = torch.mean(torch.stack(tokens_from_all_layers), dim=0).cpu().numpy()
            else:
                embedding = outputs.last_hidden_state[0][0].cpu().numpy()   # 用实体uri对应的文本句子输出的cls作为当前实体的嵌入
            node_edge_embed_dict[triple_list[i]] = embedding
    '''


final_embedding_dict = dict()
for key in embedding_dict:
    embedding = torch.mean(torch.stack(embedding_dict[key]), dim=0).cpu().numpy()
    final_embedding_dict[key] = embedding


print("\tSave embedding dict object to file by pickle ...")
with open(related_file_save_path+'embedding_dict_mp_pt_12_subgraph_triple.pkl', 'wb') as f:
    pickle.dump(final_embedding_dict, f)
print(f"End init entity and property embedding by pre-trained BERT, total entity and prop: {len(final_embedding_dict)}")

with open(related_file_save_path+'nodes.txt', 'w') as f:
    for node in nodes:
        f.write(node+'\n')
with open(related_file_save_path+'edges.txt', 'w') as f:
    for edge in edges:
        f.write(edge+'\n')