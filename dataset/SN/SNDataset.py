from torch_geometric.data import Dataset,Data
import numpy as np
import torch
import os
import random
from tqdm import tqdm
import time
from datetime import datetime
import pandas as pd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def min_date_to_timestamp(timestr):
    time_str = timestr[0:4] + "-" + timestr[4:6] + "-" + timestr[6:8] + " " + timestr[8:10] + ":" + timestr[10:12] + ":00.00"
    datetime_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return obj_stamp

def second_date_to_timestamp(timestr):
    datetime_obj = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S.%f")
    obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return obj_stamp

class SNDataset(Dataset):
    def __init__(self,root,transform=None,pre_transform=None):
        super(SNDataset, self).__init__(root,transform,pre_transform)

    @property
    def raw_file_names(self):
        tracelabelfile='traceLabel.csv'
        graphfile="trace_graph_depth10.txt"
        logfile="log_template_depth10.txt"
        vectortemplatefile="log_template_vector300_depth10.txt"
        SpanTemplateIdfile="SpanTemplateId_depth10.txt"
        service_metricfile="SN_pod_metric_with_servicelatency_normalization.csv"
        service_relation_metricfile="relation_service_kpi_normalization.csv"
        return [graphfile,logfile,vectortemplatefile,SpanTemplateIdfile,
                service_metricfile,service_relation_metricfile,tracelabelfile]

    @property
    def processed_file_names(self):
        return ['data_0_depth10.pt']

    def download(self):
        pass

    def process(self):
        edge_type_dict = {'CSCE_to_SRE': 0, 'SRE_to_CSCE': 1, 'CSCE_to_SME': 2,
                          'SME_to_CSCE': 3, 'CSCE_to_CSNE': 4, 'CSNE_to_CSCE': 5,
                          'SRE_to_SME': 6, 'SME_to_SRE': 7, 'SRE_to_CSNE': 8,
                          'CSCE_to_CSCE': 9, 'SME_to_SME': 10}
        label_dict = {'cpu_load':0,'network_loss':1,'network_delay':2}
        event_dim=300
        trace_set = set()
        with open("./OBD/raw/SN_used_traces.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for l in lines:
                l = l.replace("\n", "").strip()
                trace_set.add(l)
        f.close()

        trace_anomaly={}
        with open(self.raw_paths[6], "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "").strip().split(",")
                if line[0] == "TraceID":
                    continue
                traceid = line[0]
                anomaly_type = line[2]
                trace_anomaly[traceid]=anomaly_type
        f.close()

        service_metric={}
        df = pd.read_csv(self.raw_paths[4], index_col=0)
        for column in df.columns:
            if column.find("_networklatency") >= 0:
                continue
            for index in df.index:
                value = df.at[index, column]
                service_metric[(column, index)] = value

        service_relation_metric={}
        with open(self.raw_paths[5], "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "").strip().split(",")
                if line[0] == "Service1":
                    continue
                service_name1 = line[0]
                service_name2 = line[1]
                timestamp = min_date_to_timestamp(line[2])
                networklatency = float(line[9])
                service_relation_metric[(service_name1, service_name2, timestamp)] = networklatency
        f.close()

        template_vector_dict = {}
        SpanTemplateId=set()
        with open(self.raw_paths[3], encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.replace("\n", "").strip()
                SpanTemplateId.add(line)
        f.close()

        with open(self.raw_paths[2], encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.replace("\n", "").strip().split()
                id = line[0]
                templateVector = line[1:]
                template_vector_dict[id] = list(map(float,templateVector))
        f.close()

        logVector={}
        logType={}
        logServiceName={}
        logTimeStamp = {}
        logSpanId={}
        logTemplateId = {}
        index = 0
        with open(self.raw_paths[1], encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip().replace("\n", "").split(" - ")
                content = line[0].split("[SW_CTX:[")
                timestamp_str = content[0][:23]
                timestamp=second_date_to_timestamp(timestamp_str)
                service_name = content[1].split(",")[0].strip()
                spanid = content[1].split(",")[3].split("]")[0].strip()
                template_id = line[1]
                logId = str(index)
                logVector[logId] = template_vector_dict[template_id]
                logTimeStamp[logId] = timestamp
                logServiceName[logId] = service_name
                logTemplateId[logId] = int(template_id)
                logSpanId[logId] = spanid
                if template_id in SpanTemplateId:
                    logType[logId] = "SpanEvent"
                else:
                    logType[logId] = "LogEvent"
                index += 1
        f.close()

        idx=0
        with open(self.raw_paths[0], 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                SME = set()
                CSCE = set()
                SRE = set()
                CSNE = set()
                line = line.strip().replace("\n", "").replace("\'", "").split("           ")
                trace_id = line[0]
                if trace_id not in trace_set:
                    continue
                anomaly=trace_anomaly[trace_id]
                if anomaly.find("_")>=0:
                    fault_type=label_dict[anomaly.split("_")[1]+"_"+anomaly.split("_")[2]]
                else:
                    fault_type=3
                edges = line[1][2:-2].split("), (")
                for edge in edges:
                    edge = edge.split(", ")
                    node1 = edge[0]
                    node2 = edge[1]
                    if logType[node1] == 'SpanEvent':
                        CSCE.add(node1)
                    else:
                        SME.add(node1)
                    if logType[node2] == 'SpanEvent':
                        CSCE.add(node2)
                    else:
                        SME.add(node2)
                    node1_SRE = node1 + "_SRE"
                    node2_SRE = node2 + "_SRE"
                    SRE.add(node1_SRE)
                    SRE.add(node2_SRE)
                    if logType[node1] == 'SpanEvent' and logType[node2] == 'SpanEvent' and logServiceName[node1] != \
                            logServiceName[node2]:
                        node1_node2_CSNE = node1 + "_" + node2 + "_CSNE"
                        CSNE.add(node1_node2_CSNE)
                NodeX = []
                NodeType = []
                for logId in SME:
                    NodeX.append(logVector[logId]+[0]*3)
                    NodeType.append(0)

                for logId in CSCE:
                    NodeX.append(logVector[logId]+[0]*3)
                    NodeType.append(1)

                for nodeId in SRE:
                    logId = nodeId.split("_SRE")[0]
                    logId_timestamp = 1000 * int(logTimeStamp[logId] / 1000)
                    if (logServiceName[logId] + "_cpu", logId_timestamp) in service_metric:
                        cpu_value = service_metric[(logServiceName[logId] + "_cpu", logId_timestamp)]
                    else:
                        cpu_value = -1
                    if (logServiceName[logId] + "_memory", logId_timestamp) in service_metric:
                        memory_value = service_metric[(logServiceName[logId] + "_memory", logId_timestamp)]
                    else:
                        memory_value = -1
                    NodeX.append([0] * event_dim + [cpu_value, memory_value] + [0])
                    NodeType.append(2)

                for nodeId in CSNE:
                    logId1 = nodeId.split("_")[0]
                    logId2 = nodeId.split("_")[1]
                    logId1_timestamp = 60000 * int(logTimeStamp[logId1] / 60000)
                    if (logServiceName[logId1], logServiceName[logId2], logId1_timestamp) in service_relation_metric:
                        network_value = service_relation_metric[
                            (logServiceName[logId1], logServiceName[logId2], logId1_timestamp)]
                        NodeX.append([0] * (event_dim + 2) + [network_value])
                    elif (logServiceName[logId2], logServiceName[logId1], logId1_timestamp) in service_relation_metric:
                        network_value = service_relation_metric[
                            (logServiceName[logId2], logServiceName[logId1], logId1_timestamp)]
                        NodeX.append([0] * (event_dim + 2) + [network_value])
                    else:
                        network_value = -1
                        NodeX.append([0] * (event_dim + 2) + [network_value])
                    NodeType.append(3)

                SME = list(SME)
                CSCE = list(CSCE)
                SRE = list(SRE)
                CSNE = list(CSNE)
                AllNode = SME + CSCE + SRE + CSNE
                Pre_edge_index = []
                Tar_edge_index = []
                EdgeType = []
                for edge in edges:
                    edge = edge.split(", ")
                    node1 = edge[0]
                    node2 = edge[1]
                    node1_SRE = node1 + "_SRE"
                    node1_Id = AllNode.index(node1)
                    node2_Id = AllNode.index(node2)
                    node1_SRE_Id = AllNode.index(node1_SRE)
                    if logType[node1] == 'SpanEvent' and logType[node2] == 'SpanEvent' and logServiceName[node1] == \
                            logServiceName[node2]:
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(node2_Id)
                        EdgeType.append(edge_type_dict['CSCE_to_CSCE'])
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(node1_SRE_Id)
                        EdgeType.append(edge_type_dict['CSCE_to_SRE'])
                        Pre_edge_index.append(node1_SRE_Id)
                        Tar_edge_index.append(node2_Id)
                        EdgeType.append(edge_type_dict['SRE_to_CSCE'])
                    elif logType[node1] == 'SpanEvent' and logType[node2] == 'SpanEvent' and logServiceName[node1] != \
                            logServiceName[node2]:
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(node2_Id)
                        EdgeType.append(edge_type_dict['CSCE_to_CSCE'])
                        node1_node2_CSNE = node1 + "_" + node2 + "_CSNE"
                        CSNE_Id = AllNode.index(node1_node2_CSNE)
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(CSNE_Id)
                        EdgeType.append(edge_type_dict['CSCE_to_CSNE'])
                        Pre_edge_index.append(CSNE_Id)
                        Tar_edge_index.append(node2_Id)
                        EdgeType.append(edge_type_dict['CSNE_to_CSCE'])
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(node1_SRE_Id)
                        EdgeType.append(edge_type_dict['CSCE_to_SRE'])
                        Pre_edge_index.append(node1_SRE_Id)
                        Tar_edge_index.append(node1_Id)
                        EdgeType.append(edge_type_dict['SRE_to_CSCE'])
                        Pre_edge_index.append(node1_SRE_Id)
                        Tar_edge_index.append(CSNE_Id)
                        EdgeType.append(edge_type_dict['SRE_to_CSNE'])
                    elif logType[node1] == 'SpanEvent' and logType[node2] == 'LogEvent':
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(node2_Id)
                        EdgeType.append(edge_type_dict['CSCE_to_SME'])
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(node1_SRE_Id)
                        EdgeType.append(edge_type_dict['CSCE_to_SRE'])
                        Pre_edge_index.append(node1_SRE_Id)
                        Tar_edge_index.append(node2_Id)
                        EdgeType.append(edge_type_dict['SRE_to_SME'])
                    elif logType[node1] == 'LogEvent' and logType[node2] == 'SpanEvent':
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(node2_Id)
                        EdgeType.append(edge_type_dict['SME_to_CSCE'])
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(node1_SRE_Id)
                        EdgeType.append(edge_type_dict['SME_to_SRE'])
                        Pre_edge_index.append(node1_SRE_Id)
                        Tar_edge_index.append(node2_Id)
                        EdgeType.append(edge_type_dict['SRE_to_CSCE'])
                    elif logType[node1] == 'LogEvent' and logType[node2] == 'LogEvent':
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(node2_Id)
                        EdgeType.append(edge_type_dict['SME_to_SME'])
                        Pre_edge_index.append(node1_Id)
                        Tar_edge_index.append(node1_SRE_Id)
                        EdgeType.append(edge_type_dict['SME_to_SRE'])
                        Pre_edge_index.append(node1_SRE_Id)
                        Tar_edge_index.append(node2_Id)
                        EdgeType.append(edge_type_dict['SRE_to_SME'])
                edge_index = torch.as_tensor([Pre_edge_index, Tar_edge_index], dtype=torch.long)
                X = torch.as_tensor(NodeX, dtype=torch.float32)
                NodeType = torch.as_tensor(NodeType, dtype=torch.long)
                EdgeType = torch.as_tensor(EdgeType, dtype=torch.long)
                y = torch.as_tensor(fault_type, dtype=torch.long)
                data = Data(x=X, node_type=NodeType, edge_type=EdgeType,edge_index=edge_index, y=y)
                torch.save(data, os.path.join(self.processed_dir, 'data_{}_depth10.pt'.format(idx)))
                idx += 1
        f.close()

    def len(self) -> int:
        datalen=0
        basedir="./OBD/SN/processed"
        for file in os.listdir(basedir):
            datalen+=1
        return datalen-2

    def get(self,idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}_depth10.pt'))
        return data