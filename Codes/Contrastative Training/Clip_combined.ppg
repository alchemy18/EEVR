import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mp
import seaborn as sb
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from sklearn.metrics import classification_report
from torchmetrics import F1Score
from tqdm.autonotebook import tqdm
import itertools
import random 
import os


warnings.filterwarnings('ignore')

def csv_read(path):
    df = pd.read_csv(path)
    return df

def lowercase(data):
  temp  = data.lower()
  return temp

def stop_words(text):
  stop_words_set = set(stopwords.words('english'))
  word_tokens = word_tokenize(text)
  filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words_set]
  return " ".join(filtered_sentence)

def punctuations(data):
  no_punct=[words for words in data if words not in string.punctuation]
  words_wo_punct=''.join(no_punct)
  return words_wo_punct

def lemmatize(text):
  lemmatizer = WordNetLemmatizer()
  word_tokens = word_tokenize(text)
  lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
  return " ".join(lemmatized_text)

def preprcosess(text_data):
    text_data['Text'] = text_data['Text Description'].apply(lambda x: lemmatize(x))
    text_data['Text'] = text_data['Text'].apply(lambda x: stop_words(x))
    text_data['Text'] = text_data['Text'].apply(lambda x: lowercase(x))
    text_data['Text'] = text_data['Text'].apply(lambda x: punctuations(x))
    return text_data

def balanced_df(df):
    filtered_rows = []
    for index, row in tqdm(df.iterrows()):
        if row['arousal_category'] == 1:
            filtered_rows.append(row)

    filtered_df = pd.DataFrame(filtered_rows)
    balance_a_df = pd.concat([df, filtered_df], ignore_index=True)
    balance_a_df  = balance_a_df.reset_index(drop=True)
    return balance_a_df

class Text_Encoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=True):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name).to(torch.device("cuda"))
            
        for p in self.model.parameters():
            p.requires_grad = trainable
        
        for p in self.model.parameters():
            p.requires_grad = trainable
        


    def text_tokens(self,batch):
        text_embeddings = []
        for i in range(len(batch)):
            texts = batch[i]
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(torch.device("cuda"))
                
            # Tokenize and get embeddings
            encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(torch.device("cuda"))
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Extract embeddings from the last hidden state
            embeddings = model_output.last_hidden_state

            # Get the embeddings for the [CLS] token (the first token)
            cls_embeddings = embeddings[:, 0, :]

            # Alternatively, mean pooling the token embeddings to get sentence-level embeddings
            sentence_embeddings = torch.mean(embeddings, dim=1)

            text_embeddings.append(sentence_embeddings)

        return text_embeddings
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=100,  # Change projection_dim to 100
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, batch):
        text_embeddings = []
        for i in range(len(batch)):
            x = batch[i]
            projected = self.projection(x)
            x = self.gelu(projected)
            x = self.fc(x)
            x = self.dropout(x)
            x = x + projected
            x = self.layer_norm(x)
            text_embeddings.append(x)
        return text_embeddings
    
class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomMLP, self).__init__()

        # Define hidden layer dimensions
        hidden_dims = [50, 100]

        # Create sequential layers using nn.Linear and nn.ReLU activations
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x

    def get_hidden_embedding(self, x):
        x = self.layer1(x)
        return self.layer2(x)
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, text):
        self.data = data
        self.targets = targets
        self.text  = text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {}
        item['data'] = torch.from_numpy(self.data[index]).float()
        item['target'] = self.targets[index]
        item['text'] = self.text[index]
        return item
    
def windowed_preprocess(train_df, test_df):
    train = []
    test = []
    for index, row in train_df.iterrows():
        # Convert each row (Series) into a list of NumPy arrays
        row_as_list1 = np.array([np.array(value) for value in row.to_numpy()])
        train.append(row_as_list1)

    for index, row in test_df.iterrows():
        # Convert each row (Series) into a list of NumPy arrays
        row_as_list2 = np.array([np.array(value) for value in row.to_numpy()])
        test.append(row_as_list2)

    return train, test

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class CLIPModel(nn.Module):
    def __init__(self, mlp_input_dim, mlp_output_dim, device):
        super().__init__()
        self.text_encoder = Text_Encoder().to(device)
        self.combined_encoder = CustomMLP(mlp_input_dim, mlp_output_dim).to(device)
        self.text_projection = ProjectionHead(embedding_dim=768, projection_dim=100).to(device)
        self.device = device
        
    def combined_train(self,learning_rate, beta1, beta2 , epsilon , train_dataloader):
        criterion= nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam( self.combined_encoder.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
        optimizer.zero_grad()
        for batch in tqdm(train_dataloader):
            data = batch['data']
            target = batch['target']

            data, target = data.to(self.device), target.to(self.device)  # Move data and target to GPU
            optimizer.zero_grad()

            output = self.combined_encoder(data)

            target = target.unsqueeze(1).float()
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

    def forward(self, batch):
        data_values = batch['data']
        text_values = batch['text']

        combined_embeddings = self.combined_encoder.get_hidden_embedding(data_values).to(self.device)
        text_embeddings  = self.text_encoder.text_tokens(batch["text"])#.to(self.device)
        text_embeddings = self.text_projection(text_embeddings)#.to(self.device)
        text_embeddings = torch.stack(text_embeddings).to(self.device)
        combined_tensor = combined_embeddings

        # Calculating the Loss
        text_embeddings = text_embeddings.squeeze(1)
        logits = torch.matmul(text_embeddings, combined_tensor.T)
        # print("Logits ",logits.shape)
        combined_similarity = torch.matmul(combined_tensor, combined_tensor.T)
        input_size = text_embeddings.size(0) * text_embeddings.size(1)
        # print("Text Embedding ",text_embeddings.shape)
        texts_similarity = torch.matmul(text_embeddings, text_embeddings.T)
        targets = F.softmax((combined_similarity + texts_similarity) / 2, dim=-1) 
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device=torch.device("cuda")):
    # device = torch.device("cuda")
    model.to(device)
    loss_meter = AvgMeter()
    # print(f"train_loader: {train_loader}")
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        for key in batch.keys():
            if key != "text":
                # print(key)
                batch[key] = batch[key].to(device)
            
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        count = batch["data"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

def main():

    # Read EDA CSV
    eda_path = "../Data_files/EDA_labels.csv"
    eda_df = csv_read(eda_path)
    # HandCrafted features -> Input for the MLP based Contrastaive Learning Model
    relevant_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR']
    identifiers = ['Participant ID','Video ID','Gender','arousal_category','valence_category','taskwiselabel','three_class_label']
    eda_df  = eda_df[eda_df['CMA'] != 'Baseline']

    # Reading Text File for adding Text modality to the Data
    text_file = '../Data_files/Textdata.csv'
    text_data = pd.read_csv(text_file)
    text_data = preprcosess(text_data)

    p = {}

    pi = list(set(text_data['Participant ID'].tolist()))
    vi = list(set(text_data['Video ID'].tolist()))
    for i in pi:
        temp = {}
        df_pi = text_data[text_data['Participant ID'] == i]
        # print(len(text_data))
        for j in vi:
            df_vi =  df_pi[df_pi['Video ID'] == j]['Text'].tolist()[0]
            temp[j] = df_vi
        p[i] = temp

    eda_df['Text'] = pd.NA
    # File CSV to be used for training and evaluation 
    for index, row in tqdm(eda_df.iterrows()):
        i = row['Participant ID']
        j = row['Video ID']
        eda_df.at[index, 'Text'] = p[i][int(j[-1])]
    
    eda_df = eda_df.reset_index(drop=True)

    # Read PPG CSV
    ppg_path = "../Data_files/PPG_labels.csv"
    ppg_df = csv_read(ppg_path)

    # HandCrafted features -> Input for the MLP based Contrastaive Learning Model
    relevant_features_ppg = ['BPM', 'IBI', 'PPG_Rate_Mean', 'HRV_MedianNN',
    'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
    'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
    'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
    'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
    'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
    'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
    'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
    'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    
    identifiers = ['Participant ID','Video ID','Gender','arousal_category','valence_category','taskwiselabel','three_class_label']
    ppg_df  = ppg_df[ppg_df['CMA'] != 'Baseline']

    # Reading Text File for adding Text modality to the Data
    text_file = '/mnt/drive/home/pragyas/Pragya/Dataset/dataset/LAB/EEVR_Dataset/Ankush Code/Data/Text/Textdata.csv'
    text_data = pd.read_csv(text_file)
    text_data = preprcosess(text_data)

    p = {}

    pi = list(set(text_data['Participant ID'].tolist()))
    vi = list(set(text_data['Video ID'].tolist()))
    for i in pi:
        temp = {}
        df_pi = text_data[text_data['Participant ID'] == i]
        # print(len(text_data))
        for j in vi:
            df_vi =  df_pi[df_pi['Video ID'] == j]['Text'].tolist()[0]
            temp[j] = df_vi
        p[i] = temp

    ppg_df['Text'] = pd.NA
    # File CSV to be used for training and evaluation 
    for index, row in tqdm(ppg_df.iterrows()):
        i = row['Participant ID']
        j = row['Video ID']
        ppg_df.at[index, 'Text'] = p[i][int(j[-1])]
    
    ppg_df = ppg_df.reset_index(drop=True)



    # Read PPG + EDA CSV
    combined_df = pd.concat([ppg_df, eda_df], axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    # HandCrafted features -> Input for the MLP based Contrastaive Learning Model
    relevant_features_combined = ['BPM', 'IBI', 'PPG_Rate_Mean', 'HRV_MedianNN', 'HRV_Prc20NN',
    'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF', 'HRV_LFn',
    'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS', 'HRV_PAS',
    'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
    'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
    'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
    'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
    'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
    'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC', 'ku_eda', 'sk_eda', 'dynrange', 'slope',
    'variance', 'entropy', 'insc', 'fd_mean', 'max_scr', 'min_scr', 'nSCR',
    'meanAmpSCR', 'meanRespSCR', 'sumAmpSCR', 'sumRespSCR']
    
    identifiers = ['Participant ID','Video ID','Gender','arousal_category','valence_category','taskwiselabel','three_class_label']
    combined_df  = combined_df[combined_df['CMA'] != 'Baseline']

    combined_df = combined_df.reset_index(drop=True)
    text_file = f"Result_combined.txt"

    with open(text_file, 'w') as file:
        for participant_id in pi:
            random_seed = [111, 42, 43]
            for seed in random_seed:
                col = ["taskwiselabel", "valence_category", "arousal_category"]
                for label_id in range(3):

                    random.seed(seed)
                    os.environ['PYTHONHASHSEED'] = str(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = True

                    if col[label_id] == "arousal_category":
                        combined_df1 = balanced_df(combined_df) # balanaced df for arousal
                    else:
                        combined_df1  = combined_df

                    # device initatlization 
                    device = torch.device("cuda")

                    # defining some basic variable for the loop 
                    batch_size = 32
                    epochs = 10
                    learning_rate = 0.001
                    beta1 = 0.9 
                    beta2 = 0.999  # Default beta2 for Adam in scikit-learn
                    epsilon = 1e-8

                    input_dim =len(relevant_features_combined) # PPG Features Dimension
                    output_dim = 1 # Category of 

                    fet = relevant_features_combined # handcrafted features

                    print(f"Participant ID {participant_id}")
                    train_data = combined_df1[combined_df1['Participant ID'] != participant_id] 
                    train_data = train_data.reset_index(drop=True)
                    test_data = combined_df1[combined_df1['Participant ID'] == participant_id]
                    X_train = train_data[fet]
                    X_text = train_data['Text']
                    X_test = test_data[fet]
                    X_train, X_test = windowed_preprocess(X_train, X_test)
                    train_y = train_data[col[label_id]].to_list()

                    # Initalization of Dataset and data Loader
                    custom_dataset = CustomDataset(X_train, train_y, X_text)
                    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size)
                    model = CLIPModel(mlp_input_dim = input_dim, mlp_output_dim = output_dim, device = device)
                    params = [
                            {"params": model.text_encoder.parameters(), "lr": 1e-3},
                            {"params": model.combined_encoder.parameters(), "lr": 1e-3},
                            {"params": itertools.chain(model.text_projection.parameters()), "lr": 1e-3, "weight_decay": 1e-3}]
                    optimizer = torch.optim.AdamW(params, weight_decay=0.)
                    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.8)
                    step = "epoch"
                    best_loss = float('inf')

                    model_path = f"best_combined_{label_id}_{seed}_{participant_id}.pt"
                    # Training loop for a Clip model
                    for epoch in range(epochs):
                        print(f"Epoch: {epoch + 1}")
                        model.train()
                        model.combined_train(learning_rate, beta1, beta2 , epsilon, train_dataloader)
                        loss_meter = AvgMeter()
                        tqdm_object = tqdm(train_dataloader, total=len(train_dataloader))
                        train_loss =   train_epoch(model, train_dataloader, optimizer, lr_scheduler, step, device)
                        model.eval()
                        valid_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, step, device)
                            
                        avg = valid_loss.avg if valid_loss else float('inf')

                        if avg < best_loss:
                            best_loss = avg
                            torch.save(model.state_dict(), model_path)
                            print("Saved Combined Best Model!")

                        lr_scheduler.step(avg)
                    
                    test_y = test_data[col[label_id]].to_list()

                    if col[label_id] == "taskwiselabel":
                        label1 = "Data of high valence and " + "Positive emotion"
                        label2 = "Data of low valence and  " + "Negative emotion"
                    else:
                        label1 = "Data of High and more " + j 
                        label2 = "Data of Low and less " + j

                    label1 = lemmatize(label1)
                    label1 = stop_words(label1)
                    label1 = lowercase(label1)
                    label1 = punctuations(label1)

                    label2 = lemmatize(label2)
                    label2 = stop_words(label2)
                    label2 = lowercase(label2)
                    label2 = punctuations(label2)

                    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                    best_model = CLIPModel(mlp_input_dim = input_dim, mlp_output_dim = output_dim, device = device).to(device)
                    best_model.load_state_dict(torch.load(model_path, map_location=device))
                    pred = []
                    text_embeddings  = best_model.text_encoder.text_tokens([label1])
                    text_embeddings = best_model.text_projection(text_embeddings)
                    text_embeddings = torch.stack(text_embeddings).to(device)
                    encoded_label1 = text_embeddings.squeeze(1)
                    encoded_label1 = encoded_label1.to(device)

                    text_embeddings  = best_model.text_encoder.text_tokens([label2])
                    text_embeddings = best_model.text_projection(text_embeddings)
                    text_embeddings = torch.stack(text_embeddings).to(device)
                    encoded_label2 = text_embeddings.squeeze(1)
                    encoded_label2 = encoded_label2.to(device)

                    for i in X_test:
                        features = best_model.eda_encoder.get_hidden_embedding(torch.from_numpy(i).float().to(device))
                        a = torch.matmul(features, encoded_label1.T) 
                        b = torch.matmul(features, encoded_label2.T)
                        value = 1 if a > b else 0
                        pred.append(value)

                    print(f"Prediction: {pred}")
                    print(f"True Values: {test_y}")
                    print(classification_report(test_y, pred))

                    accuracy = accuracy_score(test_y, pred)
                    f1 = f1_score(test_y, pred)

                    print(f"Accuracy {accuracy}")
                    print(f"F1 Score {f1}")


                    # Write the result to the file
                    file.write(f"Participant {participant_id}, Label {col[label_id]}, Seed {seed}\n")
                    file.write(f"Accuracy {accuracy}\n")
                    file.write(f"F1 {f1}\n")

if __name__ == "__main__":
    main()
