# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
# In[1]:

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.preprocessing import normalize
from torchsummary import summary
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# In[1]:
embedding_size = 2048 #resnet
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps") #use Apple Silicon

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # define a transform to pre-process the images

    #mean and std of resnet50/imagenet
    rgb_mean = [0.485, 0.456, 0.406]
    std_mean = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),#resnet50 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=std_mean)])

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)

    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory
    batch_size = 256
    num_images = len(train_dataset)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16)
    #use resnet50 to get embeddings from images in train_loader
    with torch.no_grad():
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        stripped_model = nn.Sequential(*list(model.children())[:-1]) #from https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
        print(stripped_model)
        for i,batch in enumerate(train_loader):
            print("Getting embeddings on batch #%s of %s"%(i, num_images/batch_size) )
            
            features_batch = batch[0].to(device)
            embeddings_batch = stripped_model(features_batch)
            if i == 0:
                embeddings = embeddings_batch
            else:
                embeddings = torch.cat((embeddings, embeddings_batch), 0)
                    
    embeddings = embeddings.reshape(num_images, embedding_size) # flatten it to a 2d matrix
    embeddings = embeddings.cpu().numpy()
    np.save('dataset/embeddings.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings.npy')
    
    #Normalize the embeddings across the dataset
    embeddings = normalize(embeddings, norm='l2')

    #print(np.linalg.norm(embeddings[0]))

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(3*embedding_size, 1000)
        self.fc2 = nn.Linear(1000, 50)
        self.fc3 = nn.Linear(50, 150)
        self.fc4 = nn.Linear(150, 50)
        self.out = nn.Linear(50, 1)
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.out(x))
        return x.squeeze()

def train_model(train_loader, validation_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    
    summary(model, input_size=(1, 3*embedding_size))
    model.train()
    model.to(device)
    n_epochs = 1
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part 
    # of the training data as a validation split. After each epoch, compute the loss on the 
    # validation split and print it out. This enables you to see how your model is performing 
    # on the validation data before submitting the results on the server. After choosing the 
    # best model, train it on the whole training data.

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    for epoch in range(n_epochs):        
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.float().to(device) #must be float, otherwise an error!

            optimizer.zero_grad()
            y_pred = model.forward(X)

            loss = loss_function(y_pred, y)
            losses.append(loss)
            
            loss.backward()
            optimizer.step()

            if batch_idx % 400 == 0:
                print('Epoch {}, Batch idx {}, loss {}'.format(epoch, batch_idx, loss.item()))
        
        #evaluate model accuracy
        model.eval()
        with torch.no_grad():
            accuracy = 0
            for batch_idx, (X, y) in enumerate(validation_loader):
                X = X.to(device)
                y = y.float().to(device) #must be float, otherwise an error!

                y_pred = model(X)
                accuracy += accuracy_score(y.cpu(), (y_pred>=0.5).cpu())

            accuracy = (accuracy/(1+batch_idx))*100
            print("Validation set avg accuracy = %0.2f%%" % accuracy)

        

            
    return model


def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    first = True
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0

            if first:
                predictions = predicted
                first = False
            else:
                predictions = np.concatenate((predictions, predicted))


    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings.npy') == False):
        generate_embeddings()


    # load the training and testing data
    X_numpy, y_numpy = get_data(TRAIN_TRIPLETS)

    #validation set, after model selected, retrain with all data
    X_train, X_val, y_train, y_val = train_test_split(X_numpy, y_numpy, test_size=0.20, random_state=0)
     
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    print("All training data (incl val): X=%s, y=%s" % (X_numpy.shape,y_numpy.shape))
    print("Test data (which is used to generate submission): X_test=", X_test.shape)


    # In[1]:


    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X_train, y_train, train = True, batch_size=64)
    validation_loader = create_loader_from_np(X_val, y_val, train = True, batch_size=64)

    # define a model and train it
    model = train_model(train_loader, validation_loader)
    # In[1]:
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)

    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")

# %%
