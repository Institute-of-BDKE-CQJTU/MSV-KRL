## MSV-KRL

### 一、简要介绍

该项目为我们的论文“概念-属性-实例多语义视图驱动的OWL知识图谱表示学习方法”的相关代码以及数据集。



### 二、环境需求

Python版本为3.8.16，依赖包版本要求可见requirements.txt，具体如下：

```python
networkx==3.1
nltk==3.8.1
numpy==1.24.4
ordered_set==4.1.0
owlready2==0.45
rdflib==7.0.0
scikit_learn==0.24.2
torch==2.0.1
tqdm==4.65.0
transformers==4.34.1
```



### 三、数据集

HeLis、FoodOn和GO三个本体的原始文件，以及划分后的数据集，同时包括[OWL2Vec*](https://link.springer.com/article/10.1007/s10994-021-05997-6)论文中的原始数据集都位于压缩包data.zip中，本文使用三个本体的版本与[OWL2Vec\*](https://link.springer.com/article/10.1007/s10994-021-05997-6)论文中使用的版本一致。



### 四、项目整体结构

项目整体结构如下所示。其中```configs```文件夹存放有各数据集的参数配置信息；```data```文件夹存放了预训练模型、数据集以及三个本体原始文件，```bert_base```为hugging face官网下载的bert-base预训练模型，```owl2vec```文件夹存放着[OWL2Vec*项目](https://github.com/KRR-Oxford/OWL2Vec-Star)中的原始数据集，在运行代码前解压对应压缩包即可；```lib```文件夹下存放有MSV-KRL三个阶段中对应的所有代码（模型文件存放于其中的```models```目录下）;```result```文件夹用于存放各个阶段中生成的中间结果文件或者最终结果文件（请在运行代码之前手动创建该文件夹），项目训练数据集也在其中；最后```ontology_embedding.py```为运行整个项目的核心文件。

```python
MSV-KRL
  |--configs
    |--foodon_config.json
    |--go_config.json
    |--helis_config.json
  |--data
    |--bert_base
    |--owl2vec
      |--foodon
        |--test.csv
        |--train.csv
        |--valid.csv
      |--go
        |--test.csv
        |--train.csv
        |--valid.csv
      |--helis
        |--test.csv
        |--train.csv
        |--valid.csv
    |--foodon-merged.owl
    |--go.owl
    |--helis_v1.00-origin.owl
  |--lib
    |--models
      |--MTL.py
      |--PretrainBertForMLM.py
    |--__init__.py
    |--Access.py
    |--Annotations.py
    |--Bert_Dataset.py
    |--Graph.py
    |--Helper.py
    |--Knowledge_Extraction.py
    |--Label.py
    |--MTL_Datasets.py
    |--Projection.py
    |--Random_Walk.py
    |--Random.py
    |--Subgraph.py
    |--Walker.py
  |--result
    |--FoodOn
      |--bert_training_records
      |--models
      |--subgraphs
      |--triple_test_with_owl2vec.txt
      |--triple_train_with_owl2vec.txt
      |--triple_valid_with_owl2vec.txt
    |--GO
      |--bert_training_records
      |--models
      |--subgraphs
      |--triple_test_with_owl2vec.txt
      |--triple_train_with_owl2vec.txt
      |--triple_valid_with_owl2vec.txt
    |--HeLis
      |--bert_training_records
      |--models
      |--subgraphs
      |--triple_test_with_owl2vec.txt
      |--triple_train_with_owl2vec.txt
      |--triple_valid_with_owl2vec.txt
  |--ontology_embedding.py
```



### 五、运行步骤

本项目不同本体数据集需要单独进行运行，且MSV-KRL中每一个阶段需要重新配置```configs```文件夹下对应配置文件，然后单独运行。后续以HeLis本体为例，说明执行流程。

#### 1、多视图语义划分

（1）将```helis_config.json```文件中的```step_option```参数设置为```multi_views```

（2）在控制台运行```python ontology_embedding.py```命令即可，时间较长请耐心等待

#### 2、自监督进阶训练

（1）将```helis_config.json```文件中的```step_option```参数设置```fine_tune```

（2）在控制台运行```python ontology_embedding.py```命令即可，会在bert_training_records文件夹中生成训练记录

#### 3、多任务联合表示学习

（1）将```helis_config.json```文件中的```step_option```参数设置```multi_task```

（2）在控制台运行```python ontology_embedding.py```命令即可，会进行多任务的训练以及验证，由于候选实体较多，验证时间较长，请耐心等待，验证过程中以及结束时都可在控制台看见实验结果。

