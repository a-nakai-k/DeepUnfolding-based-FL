#%% 
import torch
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import pandas as pd

# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

#%%
drive_dir = "./dogs/"
splitratio = 0.8    # training data ratio
seed = 23           # random seed

filename_list = glob(drive_dir + '*/*.jpg')
tmp = []
for filename in filename_list:
    category = filename.split("/")[-2].split("-")[1]
    tmp.append([filename, category])

dog_df = pd.DataFrame(tmp, columns=['path', 'category'])
categories = dog_df['category'].unique().tolist()
dog_df['category_id'] = dog_df['category'].map(lambda x: categories.index(x))

#%%
train_df, test_df = train_test_split(dog_df, train_size=splitratio, random_state=seed)
print(train_df.shape, test_df.shape, flush=True)

#%%
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model_body = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
for param in model_body.base_model.parameters():
    param.requires_grad = False     # training only head

#%%
inputs = []
targets = []
for i in range(len(train_df)):
    data = train_df.iloc[i]
    path = data.path
    category = data.category_id
    image = Image.open(path)
    try:
        x = image_processor(image, return_tensors='pt')
        x = model_body(**x)
        x = x.last_hidden_state[:, 0, :]
        inputs.append(x)
        targets.append(category)
    except:
        pass
train_X = torch.stack(inputs)
train_y = torch.stack([torch.from_numpy(np.array(i)) for i in targets])
print(train_X.shape, flush=True)
print(train_y.shape, flush=True)

print("Train Data Processed", flush=True)

#%%
inputs = []
targets = []
for i in range(len(test_df)):
    data = test_df.iloc[i]
    path = data.path
    category = data.category_id
    image = Image.open(path)
    try:
        x = image_processor(image, return_tensors='pt')
        x = model_body(**x)
        x = x.last_hidden_state[:, 0, :]
        inputs.append(x)
        targets.append(category)
    except:
        pass
test_X = torch.stack(inputs)
test_y = torch.stack([torch.from_numpy(np.array(i)) for i in targets])
print(test_X.shape, flush=True)
print(test_y.shape, flush=True)

print("Test Data Processed", flush=True)

#%%
torch.save(train_X, 'dogs_alltrainX.pt')
torch.save(train_y, 'dogs_alltrainy.pt')
torch.save(test_X, 'dogs_alltestX.pt')
torch.save(test_y, 'dogs_alltesty.pt')

print("Preprocess Done", flush=True)