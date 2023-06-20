# Batch trainer for the Task1 model, based on lab 2.3
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch
import torch.nn.functional as F
import tqdm
import numpy as np

class Trainer():
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset ,
                 epochs: int,
                 scorer
                 ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.scorer = scorer
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.epoch_train_scores = []
        self.epoch_val_scores = []

    def run_trainer(self):
        for epoch in tqdm.tqdm(range(self.epochs)):
            self.model.train()  # train mode

            train_losses=[]
            for batch in self.training_DataLoader:
                x,y=batch
                input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
                self.optimizer.zero_grad()  # zerograd the parameters
                out = self.model(input)  # one forward pass

                loss = self.criterion(out, target)  # calculate loss
                 
                loss_value = loss.item()
                train_losses.append(loss_value)
                 
                loss.backward()  # one backward pass
                self.optimizer.step()  # update the parameters

            self.model.eval()  # evaluation mode
            valid_losses = []  # accumulate the losses here

            for batch in self.validation_DataLoader:

                x,y=batch

                with torch.no_grad():
                    out = self.model(input)   # one forward pass
                    loss = self.criterion(out, target) # calculate loss
                 
                    loss_value = loss.item()
                    valid_losses.append(loss_value)

            # save the epoch scores 
            self.epoch_train_losses.append(np.mean(train_losses))
            self.epoch_val_losses.append(np.mean(valid_losses))
            self.epoch_train_scores.append(self.get_score(self.training_DataLoader)[0])
            self.epoch_val_scores.append(self.get_score(self.validation_DataLoader)[0])
            # print the results
            print(
                f'EPOCH: {epoch+1:0>{len(str(self.epochs))}}/{self.epochs}',
                end=' '
            )
            print(f'LOSS: {np.mean(train_losses):.4f}',end=' ')
            print(f'VAL-LOSS: {np.mean(valid_losses):.4f}',end='\n')
            print(f'SCORE {self.scorer}: {self.get_score(self.training_DataLoader)[0]:.4f}',end=' ')
            print(f'VAL-SCORE {self.scorer}: {self.get_score(self.validation_DataLoader)[0]:.4f}',end='\n')

    def get_score(self, loader, scorer="default"):
        if scorer == "default":
            scorer = self.scorer

        # get the predictions
        self.model.eval()
        predictions, targets = self.get_predictions(loader)
        self.model.train()

        # Convert one-hot back to a number
        # Used in F1-score but also useful to return to generate a confusion matrix
        rounded_predictions = torch.round(predictions)

        score = 0
        match scorer:
            # case 'auc':
            #     score = roc_auc_score(targets, predictions, multi_class='ovr')
            case 'f1':
                score = f1_score(targets, rounded_predictions, average='micro')
            # case 'accuracy':
            #     score = accuracy_score(flat_targets, flat_predictions, normalize=True)
                
        return score, targets, rounded_predictions

    def get_predictions(self, loader): 
        predictions = []
        targets = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                predictions.append(outputs)
                targets.append(labels)

        predictions = torch.cat(predictions, dim=0).reshape(-1)
        targets = torch.cat(targets, dim=0).reshape(-1)
        
        return predictions, targets


