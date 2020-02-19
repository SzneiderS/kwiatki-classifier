from training.trainers import BaseTrainer
from models.base import BaseModule
from models.datasets.classifier_dataset import ClassifierDataset

import numpy as np

from torchvision.transforms import transforms

import torch
import torch.nn as nn

from visualization import TrainerVisualizer


class IrisClassifier(BaseModule):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 8)
        self.linear3 = nn.Linear(8, 3)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.prelu1(self.linear1(x))
        x = self.prelu2(self.linear2(x))
        x = self.linear3(x)
        return x

    def predict(self, x, classes=None):
        prediction = torch.argmax(self.forward(x))
        if classes is not None:
            return classes[prediction]
        return prediction

    def loss(self, output, target):
        if self.training:
            return self.loss_func(output, target.view(output.size(0)))
        else:
            return self.loss_func(output.view(1, output.size(0)), target)


if __name__ == "__main__":
    mean = torch.zeros(4)
    std = torch.zeros(4)

    transforms = {
        "input": transforms.Compose([
            transforms.Lambda(lambda x: torch.FloatTensor(x)),
            transforms.Lambda(lambda x: (x - mean) / std)
        ]),
        "target": transforms.Lambda(lambda x: torch.LongTensor([x]))
    }

    features = {}

    def process_row(row):
        for n, e in enumerate(row):
            row[n] = row[n].replace(',', '.')

        input = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]

        for i in input:  # make sure every value is > 0
            if i <= 0:
                raise ValueError("Invalid row value: {}".format(i))

        # sum features
        for i in range(0, 4):
            if i not in features:
                features[i] = []
            features[i].append(input[i])

        output = [str(row[5])]

        return input, output

    train_set, test_set, classes = ClassifierDataset.from_csv("data/iris.csv", process_row, "|",
                                                              output_type=ClassifierDataset.OutputType.INDEX,
                                                              shuffle_dataset=True, transforms=transforms)

    for i in range(0, 4):
        mean[i] = np.mean(features[i])
        std[i] = np.std(features[i])

    batch_size = 1
    for i in range(len(train_set) - 1, 2, -1):  # find biggest batch_size that divides train_set length
        if len(train_set) % i == 0:
            batch_size = i
            break

    net = IrisClassifier()
    trainer = BaseTrainer(net, train_set, test_set, batch_size=batch_size, lr=1e-2)

    def example_correct(output, target):
        return torch.argmax(output) == target

    trainer.is_example_correct_func = example_correct

    visualizer = TrainerVisualizer()
    trainer.after_epoch_func = visualizer.update_scores

    trainer.run(max_epochs=150)
    visualizer.save("plot.png")
