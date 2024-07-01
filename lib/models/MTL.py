import math
import numpy as np
import torch
import pickle
import torch.nn as nn
from tqdm import tqdm
from lib.Helper import Helper
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR




class MultiObjectLoss(nn.Module):

    def __init__(self, ema_alpha) -> None:
        super(MultiObjectLoss, self).__init__()
        # 初始化可学习的权重参数
        self.ema_alpha = ema_alpha  # 唯一超参
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.a = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.b = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.s = [0.0, 0.0, 0.0, 0.0, 0.0]

    def forward(self, outputs, labels, is_train=True):
        loss_1 = nn.BCELoss()(outputs[0][:, 0], labels[:, 0])
        loss_2 = nn.BCELoss()(outputs[1][:, 0], labels[:, 1])
        loss_3 = nn.BCELoss()(outputs[2][:, 0], labels[:, 2])
        loss_4 = nn.BCELoss()(outputs[3][:, 0], labels[:, 3])
        loss_5 = nn.BCELoss()(outputs[4][:, 0], labels[:, 4])
        losses = [loss_1, loss_2, loss_3, loss_4, loss_5]

        if is_train:
            # 通过SLAW算法更新权重
            with torch.no_grad():
                for i in range(5):
                    self.a[i] = self.ema_alpha * self.a[i] + (1 - self.ema_alpha) * (losses[i]**2)
                    self.b[i] = self.ema_alpha * self.b[i] + (1 - self.ema_alpha) * losses[i]
                    temp = math.sqrt(self.a[i] - self.b[i]**2)
                    self.s[i] = max(temp, 1e-5)
                for i in range(5):
                    summary = 1/self.s[0] + 1/self.s[1] + 1/self.s[2] + 1/self.s[3] + 1/self.s[4]
                    self.weights[i] = (5/self.s[i])/summary

        total_loss = self.weights[0]*loss_1 + self.weights[1]*loss_2 + self.weights[2]*loss_3 + self.weights[3]*loss_4 + self.weights[4]*loss_5
        
        return total_loss
    



class MTL(nn.Module):

    def __init__(self, entity_embeddings, relation_embeddings, idx2entity, idx2relation, params=None, device="cpu"):
        super(MTL, self).__init__()
        self.p = params
        self.device = device

        self.entity_embeddings = torch.tensor(entity_embeddings).to(device)
        self.relation_embeddings = torch.tensor(relation_embeddings).to(device)

        self.idx2entity = idx2entity
        self.idx2relation = idx2relation

        self.loss_func = MultiObjectLoss(self.p.ema_alpha)

        self.sub_class_pred_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.p.embed_dim*3, self.p.hidden_size),
            nn.ReLU(),
            nn.Linear(self.p.hidden_size, 1),
            nn.Sigmoid()
        )
        self.class_assert_pred_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.p.embed_dim*3, self.p.hidden_size),
            nn.ReLU(),
            nn.Linear(self.p.hidden_size, 1),
            nn.Sigmoid()
        )
        self.class_obj_prop_pred_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.p.embed_dim*3, self.p.hidden_size),
            nn.ReLU(),
            nn.Linear(self.p.hidden_size, 1),
            nn.Sigmoid()
        )
        self.obj_prop_pred_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.p.embed_dim*3, self.p.hidden_size),
            nn.ReLU(),
            nn.Linear(self.p.hidden_size, 1),
            nn.Sigmoid()
        )
        self.data_prop_pred_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.p.embed_dim*3, self.p.hidden_size),
            nn.ReLU(),
            nn.Linear(self.p.hidden_size, 1),
            nn.Sigmoid()
        )


    def forward(self, samples, neg_ents=None):
        
        triple_embeds = list()
        for sample in samples:

            sample_tensor = torch.stack([self.entity_embeddings[sample[0].item()], self.relation_embeddings[sample[1].item()], self.entity_embeddings[sample[2].item()]])

            triple_embed = sample_tensor.reshape((1, -1))[0]
            triple_embeds.append(triple_embed)
        
        triple_embeds_tensor = torch.stack(triple_embeds)     

        logits_1 = self.sub_class_pred_head(triple_embeds_tensor)
        logits_2 = self.class_assert_pred_head(triple_embeds_tensor)
        logits_3 = self.class_obj_prop_pred_head(triple_embeds_tensor)
        logits_4 = self.obj_prop_pred_head(triple_embeds_tensor)
        logits_5 = self.data_prop_pred_head(triple_embeds_tensor)

        return [logits_1, logits_2, logits_3, logits_4, logits_5]
    

    def run_training(self, train_dataloader, valid_dataloader, test_dataloader):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.p.lr, weight_decay=self.p.l2)

        sheduler = StepLR(optimizer, step_size=self.p.step_size, gamma=self.p.gamma)

        # 设定早停机参数
        patience = self.p.patience  # 如果连续5次验证集性能不提升，则触发早停机
        best_val_loss = float('inf')
        current_patience = 0

        epoch_losses = {"train":[], "valid":[]}
        for epoch in range(self.p.epoch):
            
            ## training
            self.train()
            epoch_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f'Train Epoch {epoch + 1}/{self.p.epoch}'):               

                optimizer.zero_grad()

                samples, labels = batch
                samples = torch.tensor(samples, device=self.device)
                labels = torch.tensor(labels, device=self.device)

                outputs = self.forward(samples)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_dataloader)
            epoch_losses['train'].append(avg_loss)
            print(f'Epoch {epoch + 1}/{self.p.epoch}: Loss {avg_loss:.4f}')


            ## validation
            self.eval()
            with torch.no_grad():
                valid_total_loss = 0.0
                for batch in tqdm(valid_dataloader, desc=f'Valid Epoch {epoch + 1}/{self.p.epoch}'):

                    samples, labels = batch
                    samples = torch.tensor(samples, device=self.device)
                    labels = torch.tensor(labels, device=self.device)

                    outputs = self.forward(samples)
                    loss = self.loss_func(outputs, labels, is_train=False)
                    
                    valid_total_loss += loss.item()
            
                valid_avg_loss = valid_total_loss / len(valid_dataloader)
                epoch_losses['valid'].append(valid_avg_loss)
                print(f'Epoch {epoch + 1}/{self.p.epoch}: Loss {valid_avg_loss:.4f}')


            # 判断早停
            if self.p.early_stop:
                if valid_avg_loss < best_val_loss:
                    # 保存最好的一次模型
                    torch.save(self, self.p.related_file_save_path + self.p.mtl_model_save_path + "MTL.pth")
                    best_val_loss = valid_avg_loss
                    current_patience = 0
                else:
                    current_patience += 1
                    if current_patience >= patience:
                        print("Early stopping triggered!")
                        break  # 提前停止训练
            else:
                # 不早停
                if (epoch + 1) % 5 == 0:
                    torch.save(self, self.p.related_file_save_path + self.p.mtl_model_save_path + f"MTL_{epoch+1}.pth")

                ## 无论早不早停，都保存在验证集上最优的模型
                if valid_avg_loss < best_val_loss:
                    torch.save(self, self.p.related_file_save_path +self.p.mtl_model_save_path + f"MTL_best.pth")
                    best_val_loss = valid_avg_loss


            ## 训练一轮就更新一次
            self.write_training_losses(epoch_losses)

            sheduler.step()     # 修改学习率
            

        # 返回需要的数据
        return epoch_losses
    

    def write_training_losses(self, epoch_losses):
        with open(self.p.related_file_save_path+"MTL_losses.pkl", 'wb') as f:
            pickle.dump(epoch_losses, f)

    
    def evaluate(self, evaluate_dataloader):
        self.eval()
        with torch.no_grad():
            total_triple_count = 0
            total_loss = 0.0

            metrics = {
                Helper.CLASS_SUBSUMPTION: [0, 0, 0, 0],
                Helper.CLASS_MEMBERSHIP: [0, 0, 0, 0],
                Helper.CLASS_PROPERTY: [0, 0, 0, 0],
                Helper.INDIVIDUAL_OBJECT_PROPERTY: [0, 0, 0, 0],
                Helper.INDIVIDUAL_DATA_PROPERTY: [0, 0, 0, 0]
            }
            sample_count = {
                Helper.CLASS_SUBSUMPTION: 0,
                Helper.CLASS_MEMBERSHIP: 0,
                Helper.CLASS_PROPERTY: 0,
                Helper.INDIVIDUAL_OBJECT_PROPERTY: 0,
                Helper.INDIVIDUAL_DATA_PROPERTY: 0
            }

            for batch in tqdm(evaluate_dataloader, desc=f'Evaluating '):

                for item in batch:
                    samples = torch.tensor(item[0], device=self.device)
                    ground_truth = item[1]
                    candidate_entities_idx = item[2]
                    task_type = item[3]

                    outputs = self.forward(samples)

                    task_logits = outputs[task_type - 1].clone().view(1, -1)[0].cpu().numpy()
                    max_indices = np.argsort(task_logits)[::-1]
                    max_scores = [task_logits[index] for index in max_indices]
                    max_entities = [candidate_entities_idx[index] for index in max_indices] 

                    rank = max_entities.index(ground_truth) + 1
                    metrics[task_type][0] += 1.0 / rank
                    metrics[task_type][1] += 1 if ground_truth in max_entities[:1] else 0
                    metrics[task_type][2] += 1 if ground_truth in max_entities[:5] else 0
                    metrics[task_type][3] += 1 if ground_truth in max_entities[:10] else 0
                    sample_count[task_type] += 1
                    total_triple_count += 1
                    
                    if total_triple_count % 5 == 0:
                        for key in metrics:
                            if sample_count[key] != 0:
                                print(f"Task: {Helper.task_name_dict[key]}, {sample_count[key]} triples evaluated, MRR: {metrics[key][0]/sample_count[key]}, Hits@1: {metrics[key][1]/sample_count[key]}, Hits@5: {metrics[key][2]/sample_count[key]}, Hits@10: {metrics[key][3]/sample_count[key]}")
                            else:
                                print(f"Task {Helper.task_name_dict[key]} did not have evaluated triples yet.")
                        
                        print()
            
            ## 计算全部测试样本指标
            for key in metrics:
                if sample_count[key] != 0:
                    print(f"Task: {Helper.task_name_dict[key]}, {sample_count[key]} triples evaluated, MRR: {metrics[key][0]/sample_count[key]}, Hits@1: {metrics[key][1]/sample_count[key]}, Hits@5: {metrics[key][2]/sample_count[key]}, Hits@10: {metrics[key][3]/sample_count[key]}")
                else:
                    print(f"Task {Helper.task_name_dict[key]} did not have evaluated triples yet.")
                    
