import torch
import torch.optim as optim
import time
from torch_geometric.loader import DataLoader
from Microservices.FTP.DCGFI.dataset.TT.TTDataset import TTDataset
from Microservices.FTP.DCGFI.dataset.SN.SNDataset import SNDataset
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score,recall_score
import random
from Microservices.FTP.DCGFI.utils.log import Logger
import logging
from collections import defaultdict
import pandas as pd

class DCGFITrainer():
    def __init__(self,batch_size,lr,n_epochs,model_path,dataName,out_channels,num_class,beta,warm_up_n_epochs,train_ratio):
        super().__init__()
        self.device="cuda:0"
        self.dataName=dataName
        self.out_channels = out_channels
        self.num_class = num_class
        self.beta = beta
        self.model_path=model_path
        self.log_path = self.model_path+"trainlog"
        self.logger = Logger(self.log_path, logging.INFO, __name__).getlog()
        self.train_ratio = train_ratio
        if self.dataName=='TT':
            self.dataset=TTDataset("./OBD/TT/")
            self.train_index, self.val_index, self.test_index = self.data_index_generate(dataset='TT')
        else:
            self.dataset = SNDataset("./OBD/SN/")
            self.train_index, self.val_index, self.test_index = self.data_index_generate(dataset='SN')
        self.batch_size=batch_size
        self.lr=lr
        self.n_epochs=n_epochs
        self.warm_up_n_epochs = warm_up_n_epochs
        self.label_class = {
            i: {
                'min': torch.full((out_channels,), float('inf'), device=self.device),
                'max': torch.full((out_channels,), float('-inf'), device=self.device),
                'mean': torch.zeros(out_channels, device=self.device),
            }
            for i in range(self.num_class)
        }


    def data_index_generate(self,dataset):
        if dataset=='TT':
            all_index = range(0,2242)
        else:
            all_index = range(0,1760)
        train_index = random.sample(all_index, round(self.train_ratio * len(all_index)))
        leave_index = set(all_index) - set(train_index)
        val_index = random.sample(list(leave_index), round(0.5 * len(leave_index)))
        test_index = list(set(all_index) - set(train_index) - set(val_index))
        self.logger.info("Train dataset: " + str(len(train_index)))
        self.logger.info("Val dataset: " + str(len(val_index)))
        self.logger.info("Test dataset: " + str(len(test_index)))
        return train_index, val_index, test_index


    def train(self, net):
        net = net.to(self.device)
        total_params = sum(p.numel() for p in net.parameters())
        self.logger.info(f'Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)')
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        train_dataset= self.dataset[self.train_index]
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8)
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-4)
        criterion = nn.CrossEntropyLoss().to(self.device)

        avgevents_training_time = defaultdict(list)
        self.logger.info('Starting training...')
        start_time = time.time()
        for epoch in range(self.n_epochs):
            loss_epoch = 0.0
            n_step=0
            epoch_start_time = time.time()
            net.train()
            for data in train_loader:
                batch_start_time = time.time()
                data=data.to(self.device)
                optimizer.zero_grad()
                graph_embeddings,graph_outputs = net(data)
                data_loss = criterion(graph_outputs,data.y)
                knowledge_loss = self.knowledge_loss_function(graph_embeddings,data.y)
                loss = data_loss+knowledge_loss
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_step+=1
                batch_end_time = time.time()
                batch_training_time = batch_end_time - batch_start_time
                num_nodes_per_trace = torch.bincount(data.batch)
                avg_events_per_batch = num_nodes_per_trace.float().mean().item()
                avgevents_training_time[round(avg_events_per_batch, 1)].append(batch_training_time)

            epoch_train_time = time.time() - epoch_start_time
            self.logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.10f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_step))
            torch.save(net.state_dict(), self.model_path + "/best_network.pth")

            if (epoch+1) % self.warm_up_n_epochs == 0:
                self.update_knowledge(train_loader,net)
                self.logger.info('update knowledge!!!!!!!!!!!')

            scheduler.step()

        train_time = time.time() - start_time
        self.logger.info('Training time: %.3f' % train_time)
        if torch.cuda.is_available():
            final_peak = torch.cuda.max_memory_allocated(self.device)
            self.logger.info(f"Training GPU Peak Memory: {final_peak / 1024 / 1024:.2f} MB")
        self.logger.info('Finished training.')

        event_train_timedf = pd.DataFrame(
            [{'Avg_Events': k, 'Batch_Training_Time': sum(v) / len(v)}
             for k, v in avgevents_training_time.items()]
        )
        event_train_timedf.to_csv(self.model_path+'avg_events_batch_training_time.csv', index=False)

        return net

    def validate(self,net):
        net = net.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        val_dataset = self.dataset[self.val_index]
        val_loader=DataLoader(val_dataset,batch_size=self.batch_size,shuffle=False,num_workers=8)

        self.logger.info('Starting validating...')
        pred_labels = []
        true_labels = []
        avgevents_validating_time = defaultdict(list)
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                batch_start_time = time.time()
                data = data.to(self.device)
                _, graph_outputs = net(data)
                _, preds = torch.max(graph_outputs.data, 1)
                pred_labels.append(preds)
                true_labels.append(data.y)
                batch_end_time = time.time()
                batch_valiating_time = batch_end_time - batch_start_time
                num_nodes_per_trace = torch.bincount(data.batch)
                avg_events_per_batch = num_nodes_per_trace.float().mean().item()
                avgevents_validating_time[round(avg_events_per_batch, 1)].append(batch_valiating_time)
        validate_time = time.time() - start_time
        self.logger.info('Validating time: %.3f' % validate_time)
        if torch.cuda.is_available():
            final_peak = torch.cuda.max_memory_allocated(self.device)
            self.logger.info(f"Validating GPU Peak Memory: {final_peak / 1024 / 1024:.2f} MB")

        true_labels = torch.cat(true_labels, dim=0).cpu().detach().numpy()
        pred_labels = torch.cat(pred_labels, dim=0).cpu().detach().numpy()
        macro_precision = precision_score(true_labels, pred_labels, average='macro')
        macro_recall = recall_score(true_labels, pred_labels, average='macro')
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        micro_precision = precision_score(true_labels, pred_labels, average='micro')
        micro_recall = recall_score(true_labels, pred_labels, average='micro')
        micro_f1 = f1_score(true_labels, pred_labels, average='micro')

        self.logger.info("================================================")
        self.logger.info('Macro-Precision: ' + str(macro_precision))
        self.logger.info('Macro-Recall: ' + str(macro_recall))
        self.logger.info('Macro-F1: ' + str(macro_f1))
        self.logger.info("------------------------------------------------")
        self.logger.info('Micro-Precision: ' + str(micro_precision))
        self.logger.info('Micro-Recall: ' + str(micro_recall))
        self.logger.info('Micro-F1: ' + str(micro_f1))
        self.logger.info('Finished validating.')

        event_validate_timedf = pd.DataFrame(
            [{'Avg_Events': k, 'Batch_Validating_Time': sum(v) / len(v)}
             for k, v in avgevents_validating_time.items()]
        )
        event_validate_timedf.to_csv(self.model_path+'avg_events_batch_validating_time.csv', index=False)



    def test(self, net):
        net = net.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        test_dataset = self.dataset[self.test_index]
        test_loader = DataLoader(test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=8)

        self.logger.info('Starting testing...')
        pred_labels = []
        true_labels = []
        avgevents_testing_time = defaultdict(list)
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                batch_start_time = time.time()
                data = data.to(self.device)
                _, graph_outputs = net(data)
                _, preds = torch.max(graph_outputs.data, 1)
                pred_labels.append(preds)
                true_labels.append(data.y)
                batch_end_time = time.time()
                batch_testing_time = batch_end_time - batch_start_time
                num_nodes_per_trace = torch.bincount(data.batch)
                avg_events_per_batch = num_nodes_per_trace.float().mean().item()
                avgevents_testing_time[round(avg_events_per_batch, 1)].append(batch_testing_time)

        test_time = time.time() - start_time
        self.logger.info('Testing time: %.3f' % test_time)
        if torch.cuda.is_available():
            final_peak = torch.cuda.max_memory_allocated(self.device)
            self.logger.info(f"Testing GPU Peak Memory: {final_peak / 1024 / 1024:.2f} MB")

        true_labels = torch.cat(true_labels, dim=0).cpu().detach().numpy()
        pred_labels = torch.cat(pred_labels, dim=0).cpu().detach().numpy()
        macro_precision = precision_score(true_labels, pred_labels, average='macro')
        macro_recall = recall_score(true_labels, pred_labels, average='macro')
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        micro_precision = precision_score(true_labels, pred_labels, average='micro')
        micro_recall = recall_score(true_labels, pred_labels, average='micro')
        micro_f1 = f1_score(true_labels, pred_labels, average='micro')

        self.logger.info("================================================")
        self.logger.info('Macro-Precision: ' + str(macro_precision))
        self.logger.info('Macro-Recall: ' + str(macro_recall))
        self.logger.info('Macro-F1: ' + str(macro_f1))
        self.logger.info("------------------------------------------------")
        self.logger.info('Micro-Precision: ' + str(micro_precision))
        self.logger.info('Micro-Recall: ' + str(micro_recall))
        self.logger.info('Micro-F1: ' + str(micro_f1))
        self.logger.info('Finished testing.')

        event_test_timedf = pd.DataFrame(
            [{'Avg_Events': k, 'Batch_Testing_Time': sum(v) / len(v)}
             for k, v in avgevents_testing_time.items()]
        )
        event_test_timedf.to_csv(self.model_path+'avg_events_batch_testing_time.csv', index=False)


    def update_knowledge(self, train_loader, net):
        embeddings = []
        labels = []
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                data = data.to(self.device)
                embs,outputs = net(data)
                embeddings.append(embs)
                labels.append(data.y)
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        for i in range(self.num_class):
            mask_i = labels == i
            embeddings_i = embeddings[mask_i]
            if embeddings_i.size(0) == 0:
                continue
            self.label_class[i]['min'] = torch.min(embeddings_i,dim=0).values
            self.label_class[i]['max'] = torch.max(embeddings_i, dim=0).values
            self.label_class[i]['mean'] = torch.mean(embeddings_i, dim=0)

    def knowledge_loss_function(self,embeddings,labels):
        knloss = torch.tensor(0.0, device=self.device)
        for i in range(self.num_class):
            mask_i = labels == i
            embeddings_i = embeddings[mask_i]
            if embeddings_i.size(0) == 0:
                continue
            dist_to_mean = torch.sqrt(torch.sum((embeddings_i - self.label_class[i]['mean']) ** 2, dim=1)).mean()
            dist_range = torch.sqrt(torch.sum((self.label_class[i]['max'] - self.label_class[i]['min']) ** 2))
            dist = dist_to_mean - self.beta * dist_range
            knloss += dist
        knowledge_loss = torch.clamp(knloss, min=0.0)
        return knowledge_loss