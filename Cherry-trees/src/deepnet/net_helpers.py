# Helper function to visualize the model training process
import matplotlib.pyplot as plt
from deepnet.trainer import Trainer
import torch
import torch.nn.functional as F


def plot_training_loss_progress(trainer, model_name):
    train_losses = trainer.epoch_train_losses
    val_losses = trainer.epoch_val_losses

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-o', label='Train Losses')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Losses')

    plt.title(f'{model_name} Train and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()

    plt.show()

def plot_training_score_progress(trainer, model_name):
    train_scores = trainer.epoch_train_scores
    val_scores = trainer.epoch_val_scores

    epochs = range(1, len(train_scores) + 1)

    plt.plot(epochs, train_scores, 'b-o', label='Train f1')
    plt.plot(epochs, val_scores, 'r-o', label='Validation f1')

    plt.title(f'{model_name} Train and Validation f1')
    plt.xlabel('Epochs')
    plt.ylabel('f1')
    plt.legend()

    plt.show()


# Helper function to train the models
def train_model(model, train_loader, val_loader):
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    # model
    model = model.to(device)

    # loss function
    loss = F.binary_cross_entropy

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # trainer
    trainer = Trainer(model=model,
                    device=device,
                    criterion=loss,
                    optimizer=optimizer,
                    training_DataLoader=train_loader,
                    validation_DataLoader=val_loader,
                    epochs=25,
                    scorer='f1')

    # start training
    trainer.run_trainer()
    return model, trainer   