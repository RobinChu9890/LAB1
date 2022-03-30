import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision 
from torchvision import datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings
import copy
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def images_transforms(phase):
    if phase == 'training':
        data_transformation =transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomEqualize(10),
            transforms.RandomRotation(degrees=(-25,20)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    else:
        data_transformation=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    
    return data_transformation

def imshow(img):
    plt.figure(figsize=(20, 20))
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def training(model, train_loader, test_loader, Loss, optimizer, epochs, device, num_class, name):
    model.to(device)
    best_model_wts = None
    best_evaluated_acc = 0
    train_acc = []
    test_acc = []
    test_Recall = []
    test_Precision = []
    test_F1_score = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer , gamma = 0.96)
    for epoch in range(1, epochs+1):
        with torch.set_grad_enabled(True):
            model.train()
            total_loss=0
            correct=0
            for idx,(data, label) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                        
                data = data.to(device,dtype=torch.float)
                label = label.to(device,dtype=torch.long)

                predict = model(data)      

                loss = Loss(predict, label.squeeze())

                total_loss += loss.item()
                pred = torch.max(predict,1).indices
                correct += pred.eq(label).cpu().sum().item()
                        
                loss.backward()
                optimizer.step()

            total_loss /= len(train_loader.dataset)
            correct = (correct/len(train_loader.dataset))*100.
            print ("Epoch : " , epoch)
            print ("Loss : " , total_loss)
            print ("Correct : " , correct)
            #print(epoch, total_loss, correct)     
        scheduler.step()
        accuracy, Recall, Precision, F1_score, preds, labels = evaluate(model, device, test_loader)
        train_acc.append(correct)  
        test_acc.append(accuracy)
        test_Recall.append(Recall)
        test_Precision.append(Precision)
        test_F1_score.append(F1_score)
        if accuracy > best_evaluated_acc:
            best_evaluated_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
    #save model
    torch.save(best_model_wts, name+".pt")
    model.load_state_dict(best_model_wts)

    return train_acc, test_acc, test_Recall, test_Precision, test_F1_score, preds, labels

def evaluate(model, device, test_loader):
    correct=0
    TP=0
    TN=0
    FP=0
    FN=0
    preds = []
    labels = []
    with torch.set_grad_enabled(False):
        model.eval()
        for idx,(data,label) in enumerate(test_loader):
            data = data.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.long)
            predict = model(data)
            pred = torch.max(predict,1).indices
            
            preds = preds + list(pred.cpu().numpy())
            labels = labels + list(label.cpu().numpy())
            #correct += pred.eq(label).cpu().sum().item()
            for j in range(data.size()[0]):
                #print ("{} pred label: {} ,true label:{}" .format(len(pred),pred[j],int(label[j])))
                if (int (pred[j]) == int (label[j])):
                    correct +=1
                if (int (pred[j]) == 1 and int (label[j]) ==  1):
                    TP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  0):
                    TN += 1
                if (int (pred[j]) == 1 and int (label[j]) ==  0):
                    FP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  1):
                    FN += 1
        print ("TP : " , TP)
        print ("TN : " , TN)
        print ("FP : " , FP)
        print ("FN : " , FN)

        print ("num_correct :",correct ," / " , len(test_loader.dataset))
        Recall = TP/(TP+FN)
        print ("Recall : " ,  Recall )

        Precision = TP/(TP+FP)
        print ("Precision : " ,  Precision )

        F1_score = 2 * Precision * Recall / (Precision + Recall)
        print ("F1 - score : " , F1_score)

        correct = (correct/len(test_loader.dataset))*100.
        print ("Accuracy : " , correct ,"%")

    return correct, Recall, Precision, F1_score, preds, labels


if __name__=="__main__":
    IMAGE_SIZE=(128,128)
    batch_size=128
    learning_rate = 0.01
    epochs=30
    num_classes=2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (device)

    train_path=r'C:\Users\CTCHAN\Downloads\archive\chest_xray\train'
    test_path=r'C:\Users\CTCHAN\Downloads\archive\chest_xray\test'

    trainset=datasets.ImageFolder(train_path,transform=images_transforms('train'))
    testset=datasets.ImageFolder(test_path,transform=images_transforms('test'))

    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)
    test_loader = DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2)

    model = torchvision.models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 2)

    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.001)
    train_acc, test_acc, test_Recall, test_Precision, test_F1_score, preds, labels  = training(model, train_loader, test_loader, Loss, optimizer, epochs, device, num_classes, 'CNN_chest')

#%%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn 

cm = confusion_matrix(labels, preds)
plt.figure()
sn.heatmap(cm, annot=True, cmap='Blues', fmt = 'd')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')

#%%
plt.figure()
plt.plot(train_acc)
plt.title('Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(test_acc)
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(test_F1_score)
plt.title('Test F1-score')
plt.xlabel('Epochs')
plt.ylabel('F1-score')
