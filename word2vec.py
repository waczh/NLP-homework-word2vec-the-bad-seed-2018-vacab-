import numpy as np
import jieba
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import nltk
import string
import math
import shutil
from tqdm import tqdm
from nltk.tag import pos_tag
from torch.utils.tensorboard import SummaryWriter
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser(description='Train the model of CBOW and Skip-gram')
parser.add_argument('--IO-type',default='CBOW',type=str,metavar='TP',help='chose the I/O method')
parser.add_argument('--window-size',default=5,type=int,metavar='N',help='data split window')
parser.add_argument('--clear_history',default=True,type=bool,metavar='B',help='clear training history')
args = parser.parse_args()
args.IO_type = 'CBOW'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# args.IO_type = 'Skip-gram'

class TextDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        one_input = self.data[index]
        one_ouput = self.label[index]
        return one_input, one_ouput

class Word2ec(nn.Module):
    def __init__(self, vocab_size, embed_size, cls):
        super().__init__()
        self.cls = cls
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.linear1 = nn.Linear(in_features=self.vocab_size,out_features=self.embed_size,bias=False)# 特征嵌入
        self.linear2 = nn.Linear(in_features=embed_size,out_features=vocab_size,bias=False)
        self.prob = nn.Softmax(dim=1)
    def forward(self,x):
        x_embed = self.linear1(x.squeeze())
        x = self.linear2(x_embed)
        x = self.prob(x)
        return x.unsqueeze(-1),x_embed

def preprocessing(vocab):
    # 预处理：去除标点、分词、向量化
    all_words = []
    all_processed_words = []

    for sentence in vocab:
        processed_words = []
        temp = []
        text_no_punct = sentence.translate(str.maketrans("", "", string.punctuation))  # 去除标点
        words = word_tokenize(text_no_punct)  # 分词
        words = [word.lower() for word in words]  # 转为小写
        processed_words.extend(words)
        all_processed_words.append(processed_words)
        temp.append(words)
        for word_list in temp:
            all_words.extend(word_list)
    unique_words = list(set(all_words))  # 去重词集，用于提取one-hot

    # 构建one-hot向量
    one_hot_ = np.eye(len(unique_words))
    word_one_hot_dict = {}
    for idx, word in enumerate(unique_words):
        word_one_hot_dict[word] = one_hot_[idx]

    return word_one_hot_dict, all_processed_words, len(unique_words)


def IO_generate(one_hot_dict, ordered_words, cls, unique_count): # one_hot_dict是词典，ordered_words是预处理后的语料库，cls是CBOW或skip-gram
    in_vec = None
    ou_vec = None
    word4vis_all = []

    for sentence in ordered_words:
        data_per = torch.zeros((unique_count, len(sentence)))
        for idx, word in enumerate(sentence):
            data_per[:, idx] = torch.tensor(one_hot_dict.get(word)).squeeze()  # (features, word sequence)

        x, y, word4vis = [], [], []
        for j in range(args.window_size - 1, data_per.shape[-1]):
            middle = j - math.floor(args.window_size / 2)
            sides = list(np.arange(j - args.window_size + 1, j + 1))
            sides.remove(middle)
            word4vis.append(sentence[middle])

            if cls == 'CBOW':
                x.append(data_per[:, sides])
                y.append(data_per[:, [middle]])
            elif cls == 'Skipgram':
                x.append(data_per[:, [middle]])
                y.append(data_per[:, sides])

        word4vis_all.extend(word4vis)
        if cls == 'CBOW':
            input_vec = torch.sum(torch.stack(x, dim=0), dim=2, keepdim=True)
            output_vec = torch.stack(y, dim=0)
        elif cls == 'Skipgram':
            input_vec = torch.stack(x, dim=0)
            output_vec = torch.sum(torch.stack(y, dim=0), dim=2, keepdim=True)
        if in_vec is None:
            in_vec = input_vec
        else:
            in_vec = torch.cat([in_vec, input_vec], dim=0)
        if ou_vec is None:
            ou_vec = output_vec
        else:
            ou_vec = torch.cat([ou_vec, output_vec], dim=0)
    return in_vec, ou_vec, word4vis_all


def compute_nlss(pre,gt):
    criterion = torch.nn.NLLLoss()
    gt = gt.squeeze()
    pre = pre.squeeze()
    true_index = torch.argmax(gt,dim=1,keepdim=True)
    loss_nlss = 0
    for idx in range(true_index.shape[-1]):
        loss_nlss += criterion(pre,true_index[:,idx])
    return loss_nlss


vocab_v2 = ["Later, all students and parents are celebrating outside and Emma expresses contempt for Milo to her father,who scolds her for her bitterness.",
            "Emma apologizes, but she secretly guides Milo away from the party and through the trees until they reach cliffs overlooking the sea.",
            "She slowly moves towards him and pushes him off the cliff, after stealing the Citizenship medal from him, then sneaking back to the party.",
            "Milo's body is found shortly thereafter.",
            "Emma and David watch from a distance as people desperately try to resuscitate him and his mother breaks down into hysterics.",
            "The next day, David asks Emma if she is feeling okay following what happened to Milo, but Emma acts cheerful and neglects to talk about the matter with any kind of empathy.",
            "David's sister Angela, a psychiatrist, suggests to David that Emma is in shock and will find time to grieve whenever she is ready.",
            "Chloe, a babysitter David hired, begins work.",
            "Emma notices her stealing from David's bedroom, and utilizes this fact to blackmail Chloe.",
            "Milo's funeral is held and Emma and David speak with Mr. and Mrs. Curtis.",
            "Emma feigns sadness in front of them.",
            "Mrs. Curtis asks David for photos that he took of Milo from the day of the ceremony.",
            "That night, as David peruses the photos, he notices Emma in the background of many of them looking at Milo and his medal with a contemptuous scowl."]

# sentences = ["Kage is Teacher", "Mazong is Boss", "Niuzong is Boss",
#              "Xiaobing is Student", "Xiaoxue is Student",]
# args.window_size = 3
OH_dict, processed_words, unique_count = preprocessing(vocab_v2)
print(processed_words,type(processed_words),len(processed_words))
x, y, word4vis = IO_generate(OH_dict,processed_words,args.IO_type,unique_count)
x = x.to(device)
y = y.to(device)
print(f"[{args.IO_type}] batchsize={x.shape[0]},features={x.shape[1]}, inputsize={x.shape[2]}, ouputsize={y.shape[-1]}")
print(x.shape)
trainset = TextDataset(data=x, label=y)
trainloader = DataLoader(dataset=trainset,batch_size=64,shuffle=True)
model = Word2ec(vocab_size=unique_count,embed_size=256,cls=args.IO_type)
model = model.to(device)
epochs = 500
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=epochs//3,eta_min=0,last_epoch=-1)
if args.clear_history : shutil.rmtree(''.join(args.IO_type))
writer = SummaryWriter(''.join(args.IO_type))
best_train_loss = 1e50
is_model_stored = False
for epoch in tqdm(range(epochs)):
    loss_epoch = 0
    model.train()
    for x_trian, y_train in trainloader:
        if not is_model_stored:
            writer.add_graph(model, x_trian)
            is_model_stored = True
        out, _ = model(x_trian)
        loss = compute_nlss(out,y_train)
        loss_epoch += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if loss_epoch < best_train_loss:
        best_train_loss = loss_epoch
        torch.save(model.state_dict(),'best_model.pt')
        print("model saving")
    print(f"epoch{epoch+1},loss_persample={loss_epoch/x.shape[0]}")
    writer.add_scalar('loss_per_sample/train',loss_epoch/x.shape[0],epoch)
    scheduler.step()
best_model = Word2ec(vocab_size=unique_count,embed_size=256,cls=args.IO_type).to(device)
best_model.load_state_dict(torch.load('best_model.pt'))
best_model.eval()
with torch.no_grad():
    _, embedding = best_model(x)
    word_vec = {}
    for i in range(embedding.shape[0]):
        word_vec[word4vis[i]] = embedding[i,:].detach().cpu()
    words = list(word_vec.keys())
    num_words = len(words)
    similarity_matrix = np.zeros((num_words, num_words))
    for i in range(num_words):
        for j in range(num_words):
            vector_i = word_vec[words[i]].numpy()
            vector_j = word_vec[words[j]].numpy()
            dot_product = np.dot(vector_i, vector_j)
            norm_i = np.sqrt(np.sum(vector_i ** 2))
            norm_j = np.sqrt(np.sum(vector_j ** 2))
            similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
    plt.figure(figsize=(30, 30))
    sns.heatmap(similarity_matrix, xticklabels=words, yticklabels=words, cmap="YlGnBu")
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.show()


