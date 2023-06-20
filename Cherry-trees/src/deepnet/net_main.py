import numpy as np
import os
import random as rd
from PIL import Image
import csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from neuralnet import NET
import torch
from net_helpers import train_model, plot_training_loss_progress, plot_training_score_progress


class CherryLoader:
    def __init__(self) -> None:
        self.load_all_labels()

    def load_all_labels(self):
        self.label_dict = {}
        csv_files = ["thomas_bag_0.csv", "marco_bag_1.csv", "alec_bag_3.csv", "rik_bag_4.csv"]

        # Iterate through each CSV file
        for csv_file in csv_files:
            with open(os.path.join('Labeling/labels', csv_file), "r") as file:
                reader = csv.reader(file)
                
                # Skip the header row if it exists
                next(reader)
                
                # Process each row in the CSV file
                for row in reader:
                    # Extract the key (first two columns) and value (third column)
                    key = tuple(map(float, row[:2]))
                    value = float(row[2])
                    
                    # Store the data in the dictionary
                    self.label_dict[key] = value

    def explode_data(self, histogram, n):
        histograms = [histogram]
        
        final_histograms = []

        histograms.append(np.flip(histogram, axis=0))
        histograms.append(np.flip(histogram, 1))

        for i in range(len(histograms)):
            current_hist = histograms[i]
            final_histograms.append(current_hist)
            for j in range(n):
                editable = current_hist.copy()
                for k in range(len(editable)):
                    for l in range(len(editable[k])):
                        if editable[k][l] > 4:
                            upper_bound = min(255, editable[k][l] * 2)
                            # editable[k][l] = min(rd.randint(1, upper_bound), rd.randint(1, upper_bound)) # Favor the lower values
                            editable[k][l] = rd.randint(5, upper_bound)
                final_histograms.append(editable)
        
        return final_histograms


    def get_image_grid(self, path):
        img = Image.open(path)
        histogram = np.array(img.getdata())
        histogram = histogram.reshape((32, 16))
        return histogram

    def load_all_data(self):
        folder_path = 'Labeling/data'  # Replace with the actual path to your folder

        # Get all file names in the folder
        image_names =  [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        X = []
        y = []
        for file_name in image_names:
            file_path = os.path.join(folder_path, file_name)
            hist = self.get_image_grid(file_path)

            bag = int(file_name[3])
            idx = int(file_name.split('_')[1].split('.')[0])

            if (bag,idx) not in self.label_dict:
                continue

            label = self.label_dict[bag, idx]
            exploded = self.explode_data(hist, 2)

            # if bag==0 and idx == 3:
            #     img = Image.fromarray(hist.astype(np.uint8), 'L')
            #     img.save(fr"hist.png")
            #     img = Image.fromarray(exploded[0].astype(np.uint8), 'L')
            #     img.save(fr"exploded[0].png")
            #     assert False

            X += exploded
            y += [label] * len(exploded)

        return np.array(X), np.array(y)



def make_model():
    dl = CherryLoader()
    X, y = dl.load_all_data()

    # add grey channel
    X = X.reshape(-1, 1, 32, 16)

    # Creating the data-loaders
    X_train, X_val, y_train, y_val = map(torch.tensor, train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True, stratify=y))
    
    X_train, X_val, y_train, y_val = X_train.float(), X_val.float(), y_train.float(), y_val.float()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = NET()

    model, trainer = train_model(model, train_loader, val_loader)

    plot_training_loss_progress(trainer, "Edge Confidence Model")

    plot_training_score_progress(trainer, "Edge Confidence Model")

    return model

if __name__ == '__main__':
    make_model()
    
