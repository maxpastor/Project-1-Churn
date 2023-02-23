#We need 4 arguments: num_neurons, learning_rate, droppout, num_epochs
#This script will then be passed to Sagemaker for training
import argparse
import torch
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import logging
import sys
import os
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#from tqdm import tqdm
def roc_auc(y_true, y_pred):
    
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    auc = roc_auc_score(y_true, y_pred)


    return auc

def main(args):
    train_df = pd.read_csv(args.data_dir + "/train.csv")
    validation_df = pd.read_csv(args.test_dir + "/validation.csv")
    num_features = train_df.shape[1]-1
    num_neurons = args.neurons
    dropout_value = args.dropout
    y_train = train_df.iloc[:,0]
    x_train = train_df.iloc[:, 1:]

    y_validation = validation_df.iloc[:,0]
    x_validation = validation_df.iloc[:,1:]

    class FeedForwardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(num_features, num_neurons),
                nn.LeakyReLU(),
                nn.Linear(num_neurons, num_neurons),
                nn.Dropout(p=dropout_value),
                nn.LeakyReLU(),
                nn.Linear(num_neurons, num_neurons),
                nn.Dropout(p=dropout_value),
                nn.LeakyReLU(),
                nn.Linear(num_neurons, num_neurons),
                nn.Dropout(p=dropout_value),
                nn.LeakyReLU(),
                nn.Linear(num_neurons, num_neurons),
                nn.Dropout(p=dropout_value),
                nn.LeakyReLU(),
                nn.Linear(num_neurons, num_neurons),
                nn.Dropout(p=dropout_value),
                nn.LeakyReLU(),
                nn.Linear(num_neurons, 1),
                nn.Sigmoid()
        )
        def forward(self, x: torch.tensor):
            return self.network(x)

    def train(x, y, model, opt, loss_fn, epochs=args.epochs):
        train_loss_values = []
        train_auc_values = []

        for epoch in range(epochs):
            model.train()
            pred = model(x)
            loss = loss_fn(pred, torch.unsqueeze(y, 1))
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_loss_values.append(loss.item())
            
            if epoch % 100 == 0:
                logger.info(
                    "Train Epoch: {} Loss: {:.6f} ".format(
                        epoch,
                        loss.item(),
                    )
                )
                

        return train_loss_values, train_auc_values

    def validate(x,y, model, loss_fn):
        with torch.inference_mode():
            model.eval()
            pred = model(x)
            loss = loss_fn(pred, torch.unsqueeze(y, 1))
            auc = roc_auc(y, pred)
            logger.info(
                "Test set: Average loss: {:.4f}, AUC: {}\n".format(
                    loss, auc
                )
            )
            return loss, auc
    device = torch.device("cpu")
    #Mapping our features and labels as tensors
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)

    x_validation_tensor = torch.tensor(x_validation.values, dtype=torch.float)
    y_validation_tensor = torch.tensor(y_validation.values, dtype=torch.float)

    x_train_tensor = x_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)

    x_validation_tensor = x_validation_tensor.to(device)
    y_validation_tensor = y_validation_tensor.to(device)
    model = FeedForwardModel()
    model.to(device)
    loss_fn = nn.BCELoss()
    opt = optim.Adam(params=model.parameters(), lr=args.lr)
    losses, aucs= train(x_train_tensor, y_train_tensor, model, opt, loss_fn)
    validate(x_validation_tensor, y_validation_tensor, model, loss_fn)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neurons",
        type=int,
        default=1024,
        metavar="N",
        help="The number of neurons per layer",
    )
    parser.add_argument(
        "--epochs",
        type=int, 
        default=1000,
        metavar="N",
        help="The number of training iterations before we stop",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="Learning rate of the model (default 0.001)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        metavar="DP",
        help="The dropout applied to the model, helps reduce overfitting",
    )
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_VALIDATE"])
    main(parser.parse_args())