import torch
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils.undirected import to_undirected
import numpy as np
import random
import pandas as pd
from decimal import Decimal, InvalidOperation

def safe_convert(value):
    """安全转换字符串为整数（兼容科学计数法）"""
    try:
        # 先尝试Decimal高精度转换
        return int(Decimal(value.strip()))
    except (InvalidOperation, ValueError):
        try:
            # 回退到float转换（兼容更多格式但可能有精度损失）
            return int(float(value))
        except:
            # 无法转换时返回None（或可改为raise异常）
            return None
        

class nba(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['nba']

        super(nba, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['nba.csv','nba_relationship.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        with open(self.raw_paths[0], 'r') as f:                   
            raw_data = f.read().split('\n') 
            #filtered_data = [r for r in data if int(r.split(',')[1]) >= 0]
            header = raw_data[0].split(',')
            country_index = header.index('country')
            print('country_index:',country_index)

            data = raw_data[1:-1]
            first_column = [r.split(',')[0] for r in data] 
            idx_map = {value: idx for idx, value in enumerate(first_column)} 

            #filtered_data = [r for r in data if int(r.split(',')[1]) >= 0]

            #x = [[float(v) for v in r.split(',')[2:]] for r in filtered_data]
            x = [[float(v) for v in r.split(',')[2:]] for r in data]
            #x = [[float(v) for i, v in enumerate(r.split(',')[2:]) if i != 37 - 2] for r in data]
            x = torch.tensor(x, dtype=torch.float)
            print(x.size(0))  

            #first_column = [r.split(',')[0] for r in filtered_data]
            #new_idx_map = {idx: new_idx for new_idx, idx in enumerate(first_column)}
   
            #y = [int(r.split(',')[1]) for r in filtered_data]
            y = [int(r.split(',')[1]) for r in data]
            y = torch.tensor(y, dtype=torch.long)


        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            data = [[idx_map.get(str(v), v) if str(v) in idx_map else v for v in r] for r in data]
            
            #data = [[new_idx_map.get(str(v), v) for v in r if str(v) in new_idx_map] for r in data if all(str(v) in new_idx_map for v in r)]  

            #filtered_edges = [[v[0], v[1]] for v in data if ((v[0] <= 313) and (v[1] <= 313))]

            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            #print('edge_index',edge_index)
            edge_index = to_undirected(edge_index)
            #print('to_undirected(edge_index)',edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
            #print(edge_index)

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def sens(self):
        with open(self.raw_paths[0], 'r') as f:
            raw_data = f.read().split('\n')
            data = raw_data[1:-1] 
            #filtered_data = [r for r in data if int(r.split(',')[1]) >= 0]
            
            header = raw_data[0].split(',')

            country_index = header.index('country')
            #print('country_index:',country_index)

            sens = [int(r.split(',')[country_index]) for r in data if r]

            #sens = torch.tensor(sens, dtype=torch.float).unsqueeze(1)
            sens = torch.tensor(sens, dtype=torch.float)
            print(len(sens))
        return sens

    def get_idx(self,predict_attr,seed):
        idx_features_labels = pd.read_csv(self.raw_paths[0])

        labels = idx_features_labels[predict_attr].values

        labels = torch.LongTensor(labels)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        random.seed(seed)
        label_idx = np.where(labels>=0)[0]
        #print('Before shuffle: ',label_idx)
        random.shuffle(label_idx)
        #print('After shuffle: ',label_idx)
        return label_idx


    def __repr__(self):
        return '{}()'.format(self.name)


class pokec(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['pokec-z','pokec-n']

        super(pokec, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        if self.name =='pokec-z':
            return ['region_job.csv','region_job_relationship.txt']
        else:
            return ['region_job_2.csv','region_job_2_relationship.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        with open(self.raw_paths[0], 'r') as f:                   
            raw_data = f.read().split('\n') 

            header = raw_data[0].split(',')

            region_index = header.index('region')
            print('region_index:',region_index)

            data = raw_data[1:-1]

            first_column = [r.split(',')[0] for r in data]  
            idx_map = {value: idx for idx, value in enumerate(first_column)} 

            x = [[float(r.split(',')[i]) for i in range(len(r.split(','))) if i != 0 and i != 6] for r in data]
            x = torch.tensor(x, dtype=torch.float)
            print(x.size(0)) 

            y = [int(r.split(',')[6]) for r in data]
            y = torch.tensor(y, dtype=torch.long)


        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            data = [[idx_map.get(str(v), v) if str(v) in idx_map else v for v in r] for r in data]

            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            #print('edge_index',edge_index)
            edge_index = to_undirected(edge_index)
            #print('to_undirected(edge_index)',edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
            #print(edge_index)

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def sens(self):
        with open(self.raw_paths[0], 'r') as f:
            raw_data = f.read().split('\n')
            data = raw_data[1:-1] 
            
            header = raw_data[0].split(',')
            region_index = header.index('region')
            sens = [int(r.split(',')[region_index]) for r in data if r]
            sens = torch.tensor(sens, dtype=torch.float)
        return sens

    def get_idx(self,predict_attr,seed):
        idx_features_labels = pd.read_csv(self.raw_paths[0])

        labels = idx_features_labels[predict_attr].values

        labels = torch.LongTensor(labels)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        random.seed(seed)
        label_idx = np.where(labels>=0)[0]
        #print('Before shuffle: ',label_idx)
        random.shuffle(label_idx)
        #print('After shuffle: ',label_idx)
        return label_idx

    def __repr__(self):
        return '{}()'.format(self.name)





class german(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['german']

        super(german, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['german.csv','german_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        with open(self.raw_paths[0], 'r') as f:                   
            raw_data = f.read().split('\n') 
            #filtered_data = [r for r in data if int(r.split(',')[1]) >= 0]
            header = raw_data[0].split(',')
            gender_index = header.index('Gender')
            print('gender_index:',gender_index)

            data = raw_data[1:]     #去除第一行，列名

            print("raw_data 的类型:", type(data))
            print("第一行的类型:", type(data[0]))

            # (1) 提取第7列的所有唯一值，因为第7列是字符串
            unique_strings = list({row.split(',')[6] for row in data})  # 使用集合去重

            # (2) 创建映射字典：{"Apple": 0, "Banana": 1, ...}
            str_to_num = {s: i for i, s in enumerate(unique_strings)}

            # (3) 替换第7列的字符串为数值编码
            for i in range(len(data)):
                parts = data[i].split(',')  # 拆分当前行
                parts[6] = str(str_to_num[parts[6]])  # 修改第7列
                data[i] = ','.join(parts)  # 重新拼接并替换原数据
            
            
            print("第7列的类型:", type(data[0][6]))

            x = [[float(v) for v in r.split(',')[1:]] for r in data]
            x = torch.tensor(x, dtype=torch.float)
            print(x.size(0))

            y_raw = [int(r.split(',')[0]) for r in data]
            y = [x if x >= 0 else 0 for x in y_raw]
            y = torch.tensor(y, dtype=torch.long)


        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[int(float(v)) for v in r.split(' ')] for r in data]
            
            #data = [[new_idx_map.get(str(v), v) for v in r if str(v) in new_idx_map] for r in data if all(str(v) in new_idx_map for v in r)]  

            #filtered_edges = [[v[0], v[1]] for v in data if ((v[0] <= 313) and (v[1] <= 313))]

            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            #print('edge_index',edge_index)
            edge_index = to_undirected(edge_index)
            #print('to_undirected(edge_index)',edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
            #print(edge_index)

        data = Data(x=x, edge_index=edge_index.long(), y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def sens(self):
        with open(self.raw_paths[0], 'r') as f:
            raw_data = f.read().split('\n')
            data = raw_data[1:] 
            
            header = raw_data[0].split(',')

            gender_index = header.index('Gender')



            # (1) 提取第1列的所有唯一值，因为第1列是字符串
            unique_strings = list({row.split(',')[1] for row in data})  # 使用集合去重

            # (2) 创建映射字典：{"Apple": 0, "Banana": 1, ...}
            str_to_num = {s: i for i, s in enumerate(unique_strings)}

            # (3) 替换第1列的字符串为数值编码
            for i in range(len(data)):
                parts = data[i].split(',')  # 拆分当前行
                parts[1] = str(str_to_num[parts[1]])  # 修改第1列
                data[i] = ','.join(parts)  # 重新拼接并替换原数据
            
            
            print("第1列的类型:", type(data[0][1]))


            sens = [int(r.split(',')[gender_index]) for r in data if r]

            #sens = torch.tensor(sens, dtype=torch.float).unsqueeze(1)
            sens = torch.tensor(sens, dtype=torch.float)
            print(len(sens))
        return sens

    def get_idx(self,predict_attr,seed):
        idx_features_labels = pd.read_csv(self.raw_paths[0])

        labels = idx_features_labels[predict_attr].values

        labels = torch.LongTensor(labels)
        labels[labels < 0] = 0
        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        random.seed(seed)
        label_idx = np.where(labels>=0)[0]
        #print('Before shuffle: ',label_idx)
        random.shuffle(label_idx)
        #print('After shuffle: ',label_idx)
        return label_idx


    def __repr__(self):
        return '{}()'.format(self.name)




class bail(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['bail']

        super(bail, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['bail.csv','bail_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        with open(self.raw_paths[0], 'r') as f:                   
            raw_data = f.read().split('\n') 

            data = raw_data[1:]     #去除第一行，列名

            x = [[float(r.split(',')[i]) for i in range(len(r.split(','))) if i != 16] for r in data]
            x = torch.tensor(x, dtype=torch.float)
            print(x.size(0))

            y = [int(r.split(',')[16]) for r in data]
            y = torch.tensor(y, dtype=torch.long)


        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[int(float(v)) for v in r.split(' ')] for r in data]
            
            #data = [[new_idx_map.get(str(v), v) for v in r if str(v) in new_idx_map] for r in data if all(str(v) in new_idx_map for v in r)]  

            #filtered_edges = [[v[0], v[1]] for v in data if ((v[0] <= 313) and (v[1] <= 313))]

            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            #print('edge_index',edge_index)
            edge_index = to_undirected(edge_index)
            #print('to_undirected(edge_index)',edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
            #print(edge_index)

        data = Data(x=x, edge_index=edge_index.long(), y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def sens(self):
        with open(self.raw_paths[0], 'r') as f:
            raw_data = f.read().split('\n')
            data = raw_data[1:] 
            
            header = raw_data[0].split(',')

            gender_index = header.index('WHITE')



            sens = [int(r.split(',')[gender_index]) for r in data if r]

            #sens = torch.tensor(sens, dtype=torch.float).unsqueeze(1)
            sens = torch.tensor(sens, dtype=torch.float)
            print(len(sens))
        return sens

    def get_idx(self,predict_attr,seed):
        idx_features_labels = pd.read_csv(self.raw_paths[0])

        labels = idx_features_labels[predict_attr].values

        labels = torch.LongTensor(labels)
        labels[labels < 0] = 0
        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        random.seed(seed)
        label_idx = np.where(labels>=0)[0]
        #print('Before shuffle: ',label_idx)
        random.shuffle(label_idx)
        #print('After shuffle: ',label_idx)
        return label_idx


    def __repr__(self):
        return '{}()'.format(self.name)
    


class credit(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['credit']

        super(credit, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['credit.csv','credit_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        with open(self.raw_paths[0], 'r') as f:                   
            raw_data = f.read().split('\n') 

            data = raw_data[1:]     #去除第一行，列名


            x = [[float(v) for v in r.split(',')[1:]] for r in data]
            x = torch.tensor(x, dtype=torch.float)
            print(x.size(0))

            y_raw = [int(r.split(',')[0]) for r in data]
            y = [x if x >= 0 else 0 for x in y_raw]
            y = torch.tensor(y, dtype=torch.long)


        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[int(float(v)) for v in r.split(' ')] for r in data]
            
            #data = [[new_idx_map.get(str(v), v) for v in r if str(v) in new_idx_map] for r in data if all(str(v) in new_idx_map for v in r)]  

            #filtered_edges = [[v[0], v[1]] for v in data if ((v[0] <= 313) and (v[1] <= 313))]

            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            #print('edge_index',edge_index)
            edge_index = to_undirected(edge_index)
            #print('to_undirected(edge_index)',edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
            #print(edge_index)

        data = Data(x=x, edge_index=edge_index.long(), y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def sens(self):
        with open(self.raw_paths[0], 'r') as f:
            raw_data = f.read().split('\n')
            data = raw_data[1:] 
            
            header = raw_data[0].split(',')

            gender_index = header.index('Age')



            sens = [int(r.split(',')[gender_index]) for r in data if r]

            #sens = torch.tensor(sens, dtype=torch.float).unsqueeze(1)
            sens = torch.tensor(sens, dtype=torch.float)
            print(len(sens))
        return sens

    def get_idx(self,predict_attr,seed):
        idx_features_labels = pd.read_csv(self.raw_paths[0])

        labels = idx_features_labels[predict_attr].values

        labels = torch.LongTensor(labels)
        labels[labels < 0] = 0
        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        random.seed(seed)
        label_idx = np.where(labels>=0)[0]
        #print('Before shuffle: ',label_idx)
        random.shuffle(label_idx)
        #print('After shuffle: ',label_idx)
        return label_idx


    def __repr__(self):
        return '{}()'.format(self.name)








def DataLoader(name):
    name = name.lower()

    if name in ['nba']:
        dataset = nba(root='./data/', name=name, transform=T.NormalizeFeatures())

    elif name in ['pokec-z','pokec-n']:
        dataset = pokec(root='./data/', name=name, transform=T.NormalizeFeatures())

    elif name in ['german']:
        dataset = german(root='./data/', name=name, transform=T.NormalizeFeatures())
    
    elif name in ['bail']:
        dataset = bail(root='./data/', name=name, transform=T.NormalizeFeatures())

    elif name in ['credit']:
        dataset = credit(root='./data/', name=name, transform=T.NormalizeFeatures())

    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    return dataset
