from torch_geometric.data import Dataset,Data
import numpy as np
import torch
import os
import random
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def date_to_timestamp(timestr):
    datetime_obj = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S.%f")
    obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return obj_stamp

class TTDataset(Dataset):
    def __init__(self,root,transform=None,pre_transform=None):
        super(TTDataset, self).__init__(root,transform,pre_transform)

    @property
    def raw_file_names(self):
        graphfile1="F1_trace_graph_depth10.txt"
        graphfile2 = "F2_trace_graph_depth10.txt"
        graphfile3 = "F3_trace_graph_depth10.txt"
        graphfile4 = "Normalbig_trace_graph_depth10.txt"
        logfile1="F1_log_template_depth10.txt"
        logfile2 = "F2_log_template_depth10.txt"
        logfile3 = "F3_log_template_depth10.txt"
        logfile4 = "Normalbig_log_template_depth10.txt"
        F1_service_metricfile="F1_pod_metric_with_servicelatency_normalization.csv"
        F1_service_relation_metricfile="F1_relation_service_kpi_normalization.csv"
        F2_service_metricfile="F2_pod_metric_with_servicelatency_normalization.csv"
        F2_service_relation_metricfile="F2_relation_service_kpi_normalization.csv"
        F3_service_metricfile="F3_pod_metric_with_servicelatency_normalization.csv"
        F3_service_relation_metricfile="F3_relation_service_kpi_normalization.csv"
        normal_service_metricfile="normal_pod_metric_with_servicelatency_normalization.csv"
        normal_service_relation_metricfile="normal_relation_service_kpi_normalization.csv"
        vectortemplatefile="log_template_vector300_depth10.txt"
        SpanTemplateIdfile="SpanTemplateId_depth10.txt"
        return [graphfile1,graphfile2,graphfile3,graphfile4,
                logfile1,logfile2,logfile3,logfile4,
                F1_service_metricfile,F1_service_relation_metricfile,
                F2_service_metricfile,F2_service_relation_metricfile,
                F3_service_metricfile, F3_service_relation_metricfile,
                normal_service_metricfile, normal_service_relation_metricfile,
                vectortemplatefile,SpanTemplateIdfile]

    @property
    def processed_file_names(self):
        return ['data_0_depth10.pt']

    def download(self):
        pass

    def process(self):
        edge_type_dict = {'CSCE_to_SRE': 0, 'SRE_to_CSCE': 1, 'CSCE_to_SME': 2,
                          'SME_to_CSCE': 3, 'CSCE_to_CSNE': 4, 'CSNE_to_CSCE':5,
                          'SRE_to_SME':6, 'SME_to_SRE':7, 'SRE_to_CSNE':8,
                          'CSCE_to_CSCE':9,'SME_to_SME':10}
        event_dim = 300
        template_vector_dict = {}
        SpanTemplateId=set()
        trace_set = set()
        trace_label_dict = {}
        with open("./OBD/raw/TT_used_traces.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for l in lines:
                l = l.replace("\n", "").split()
                traceid = l[0]
                label = int(l[1])
                trace_set.add(traceid)
                trace_label_dict[traceid]=label
        f.close()

        with open(self.raw_paths[17], encoding='utf-8-sig') as f:
            for line in tqdm(f):
                line = line.replace("\n", "").strip()
                SpanTemplateId.add(line)
        f.close()

        with open(self.raw_paths[16], encoding='utf-8-sig') as f:
            for line in tqdm(f):
                line = line.replace("\n", "").strip().split()
                id = line[0]
                templateVector = line[1:]
                template_vector_dict[id] = list(map(float,templateVector))
        f.close()

        service_metric={}
        service_relation_metric={}
        for i in tqdm(range(4)):
            targetIndex=8+2*i
            df = pd.read_csv(self.raw_paths[targetIndex], index_col=0)
            for column in df.columns:
                if column.find("_networklatency")>=0 or column.find("_servicelatency")>=0 :
                    continue
                for index in df.index:
                    value=df.at[index,column]
                    service_metric[(column,index)]=value
            with open(self.raw_paths[targetIndex+1],"r",encoding='utf-8') as f:
                lines=f.readlines()
                for line in lines:
                    line = line.replace("\n","").strip().split(",")
                    if line[0]=="Service1_ID":
                        continue
                    service_name1=line[0]
                    service_name2=line[1]
                    timestamp=int(line[2])
                    networklatency=float(line[-1])
                    service_relation_metric[(service_name1,service_name2,timestamp)]=networklatency
            f.close()

        logfilelist=self.raw_paths[4:8]
        logVector={}
        logType={}
        logServiceName={}
        logTimeStamp={}
        logSpanId={}
        for i in tqdm(range(len(logfilelist))):
            index=0
            with open(logfilelist[i], encoding='utf-8') as f:
                for line in f:
                    line = line.strip().replace("\n", "").split(" - ")
                    content=line[0].split("       ")[2].split("[SW_CTX:[")
                    timestamp=date_to_timestamp(content[0][:23])
                    service_name=content[1].split(",")[0].strip()
                    spanid = content[1].split(",")[3].strip()+"S"+content[1].split(",")[4].split("]")[0].strip()
                    template_id = line[-2]
                    logId= str(i)+":"+str(index)
                    logVector[logId]=template_vector_dict[template_id]
                    logTimeStamp[logId]=timestamp
                    logServiceName[logId]=service_name
                    logSpanId[logId]=spanid
                    if template_id in SpanTemplateId:
                        logType[logId]="SpanEvent"
                    else:
                        logType[logId]="LogEvent"
                    index+=1
            f.close()

        graphfilelist=self.raw_paths[:4]
        idx=0
        for i in tqdm(range(len(graphfilelist))):
            with open(graphfilelist[i], 'r', encoding='utf-8') as f:
                for line in f:
                    SME = set()
                    CSCE=set()
                    SRE = set()
                    CSNE=set()
                    line = line.strip().replace("\n", "").replace("\'", "").split("           ")
                    traceid = line[0]
                    if traceid not in trace_set:
                        continue
                    trace_label=trace_label_dict[traceid]
                    edges = line[1][2:-2].split("), (")
                    for edge in edges:
                        edge = edge.split(", ")
                        node1 = str(i) + ":" + edge[0]
                        node2 = str(i) + ":" + edge[1]
                        if logType[node1]=='SpanEvent':
                            CSCE.add(node1)
                        else:
                            SME.add(node1)
                        if logType[node2]=='SpanEvent':
                            CSCE.add(node2)
                        else:
                            SME.add(node2)
                        node1_SRE=node1+"_SRE"
                        node2_SRE = node2 + "_SRE"
                        SRE.add(node1_SRE)
                        SRE.add(node2_SRE)
                        if logType[node1]=='SpanEvent' and logType[node2]=='SpanEvent' and logServiceName[node1]!=logServiceName[node2]:
                            node1_node2_CSNE=node1+"_"+ node2+"_CSNE"
                            CSNE.add(node1_node2_CSNE)
                    NodeX = []
                    NodeType=[]
                    for logId in SME:
                        NodeX.append(logVector[logId]+[0]*3)
                        NodeType.append(0)
                    for logId in CSCE:
                        NodeX.append(logVector[logId]+[0]*3)
                        NodeType.append(1)
                    for nodeId in SRE:
                        logId = nodeId.split("_SRE")[0]
                        logId_timestamp = 60000 * int(logTimeStamp[logId] / 60000)
                        if (logServiceName[logId] + "_cpu",logId_timestamp) in service_metric:
                            cpu_value = service_metric[(logServiceName[logId] + "_cpu",logId_timestamp)]
                        else:
                            cpu_value = -1
                        if (logServiceName[logId] + "_memory",logId_timestamp) in service_metric:
                            memory_value = service_metric[(logServiceName[logId] + "_memory",logId_timestamp)]
                        else:
                            memory_value = -1
                        NodeX.append([0]*event_dim+[cpu_value,memory_value]+[0])
                        NodeType.append(2)
                    for nodeId in CSNE:
                        logId1 = nodeId.split("_")[0]
                        logId2 = nodeId.split("_")[1]
                        logId1_timestamp=60000 * int(logTimeStamp[logId1]/60000)
                        if (logServiceName[logId1],logServiceName[logId2],logId1_timestamp) in service_relation_metric:
                            network_value = service_relation_metric[(logServiceName[logId1],logServiceName[logId2],logId1_timestamp)]
                            NodeX.append([0]*(event_dim+2) + [network_value])
                        elif (logServiceName[logId2],logServiceName[logId1],logId1_timestamp) in service_relation_metric:
                            network_value = service_relation_metric[(logServiceName[logId2], logServiceName[logId1], logId1_timestamp)]
                            NodeX.append([0]*(event_dim+2) + [network_value])
                        else:
                            network_value = -1
                            NodeX.append([0]*(event_dim+2) + [network_value])
                        NodeType.append(3)

                    SME = list(SME)
                    CSCE=list(CSCE)
                    SRE=list(SRE)
                    CSNE=list(CSNE)
                    AllNode = SME + CSCE + SRE + CSNE
                    Pre_edge_index=[]
                    Tar_edge_index = []
                    EdgeType=[]
                    for edge in edges:
                        edge = edge.split(", ")
                        node1 = str(i) + ":" + edge[0]
                        node2 = str(i) + ":" + edge[1]
                        node1_SRE = node1 + "_SRE"
                        node1_Id = AllNode.index(node1)
                        node2_Id = AllNode.index(node2)
                        node1_SRE_Id = AllNode.index(node1_SRE)
                        if logType[node1]=='SpanEvent' and logType[node2]=='SpanEvent' and logServiceName[node1]==logServiceName[node2]:
                            Pre_edge_index.append(node1_Id)
                            Tar_edge_index.append(node2_Id)
                            EdgeType.append(edge_type_dict['CSCE_to_CSCE'])
                            Pre_edge_index.append(node1_Id)
                            Tar_edge_index.append(node1_SRE_Id)
                            EdgeType.append(edge_type_dict['CSCE_to_SRE'])
                            Pre_edge_index.append(node1_SRE_Id)
                            Tar_edge_index.append(node2_Id)
                            EdgeType.append(edge_type_dict['SRE_to_CSCE'])
                        elif logType[node1]=='SpanEvent' and logType[node2]=='SpanEvent' and logServiceName[node1]!=logServiceName[node2]:
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
                        elif logType[node1]=='SpanEvent' and logType[node2]=='LogEvent':
                            Pre_edge_index.append(node1_Id)
                            Tar_edge_index.append(node2_Id)
                            EdgeType.append(edge_type_dict['CSCE_to_SME'])
                            Pre_edge_index.append(node1_Id)
                            Tar_edge_index.append(node1_SRE_Id)
                            EdgeType.append(edge_type_dict['CSCE_to_SRE'])
                            Pre_edge_index.append(node1_SRE_Id)
                            Tar_edge_index.append(node2_Id)
                            EdgeType.append(edge_type_dict['SRE_to_SME'])
                        elif logType[node1]=='LogEvent' and logType[node2]=='SpanEvent':
                            Pre_edge_index.append(node1_Id)
                            Tar_edge_index.append(node2_Id)
                            EdgeType.append(edge_type_dict['SME_to_CSCE'])
                            Pre_edge_index.append(node1_Id)
                            Tar_edge_index.append(node1_SRE_Id)
                            EdgeType.append(edge_type_dict['SME_to_SRE'])
                            Pre_edge_index.append(node1_SRE_Id)
                            Tar_edge_index.append(node2_Id)
                            EdgeType.append(edge_type_dict['SRE_to_CSCE'])
                        elif logType[node1]=='LogEvent' and logType[node2]=='LogEvent':
                            Pre_edge_index.append(node1_Id)
                            Tar_edge_index.append(node2_Id)
                            EdgeType.append(edge_type_dict['SME_to_SME'])
                            Pre_edge_index.append(node1_Id)
                            Tar_edge_index.append(node1_SRE_Id)
                            EdgeType.append(edge_type_dict['SME_to_SRE'])
                            Pre_edge_index.append(node1_SRE_Id)
                            Tar_edge_index.append(node2_Id)
                            EdgeType.append(edge_type_dict['SRE_to_SME'])
                    edge_index=torch.as_tensor([Pre_edge_index,Tar_edge_index],dtype=torch.long)
                    X=torch.as_tensor(NodeX,dtype=torch.float32)
                    NodeType=torch.as_tensor(NodeType,dtype=torch.long)
                    EdgeType= torch.as_tensor(EdgeType,dtype=torch.long)
                    y=torch.as_tensor(trace_label,dtype=torch.long)
                    data = Data(x=X,node_type=NodeType,edge_type=EdgeType,edge_index=edge_index, y=y)
                    torch.save(data, os.path.join(self.processed_dir, 'data_{}_depth10.pt'.format(idx)))
                    idx+=1
            f.close()

    def len(self) -> int:
        datalen=0
        basedir="./OBD/TT/processed"
        for file in os.listdir(basedir):
            datalen+=1
        return datalen-2

    def get(self,idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}_depth10.pt'))
        return data