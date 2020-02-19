from . import GenericDataset
import csv
from utils import one_hot_vector
from enum import Enum
import torch


class ClassifierDataset(GenericDataset):
    def __init__(self, transforms=None):
        super(ClassifierDataset, self).__init__(transforms)

    class OutputType(Enum):
        ONE_HOT = 1
        INDEX = 2

    @staticmethod
    def _encode_output(examples, classes, output_type):
        classes_list = list(classes)
        num_of_classes = len(classes_list)
        for n, example in enumerate(examples):
            output = None

            if output_type == ClassifierDataset.OutputType.ONE_HOT:
                indices = []
                for example_class in example[1]:
                    indices.append(classes_list.index(example_class))
                output = one_hot_vector(num_of_classes, indices)
            if output_type == ClassifierDataset.OutputType.INDEX:
                output = classes_list.index(example[1][0]) #there can be only one class for INDEX output type

            examples[n] = (example[0], output)
        return examples

    @staticmethod
    def _class_percentage(dataset):
        classes = {}
        for e in dataset:
            if str(e[1]) not in classes:
                classes[str(e[1])] = 0
            classes[str(e[1])] += 1
        classes = [a / len(dataset) for _, a in sorted(classes.items())]
        return classes

    @staticmethod
    def _classes_equal_perc(dataset):
        percentages = ClassifierDataset._class_percentage(dataset)
        equal = 1 / len(percentages)
        for e in percentages:
            if abs(e - equal) > 0.03:
                return False
        return True


    @classmethod
    def from_csv(cls, csv_filename, process_row_cb, delimiter=' ', ignored_first_rows=1, output_type=OutputType.ONE_HOT,
                 equalize_classes=True, **kwargs):
        with open(csv_filename, 'r') as csv_file:
            examples = []
            reader = csv.reader(csv_file, delimiter=delimiter)
            classes = set()
            for n, row in enumerate(reader):
                try:
                    if n < ignored_first_rows:
                        continue

                    inp, out = process_row_cb(row)

                    for c in out:
                        classes.add(c)

                    examples.append((inp, out))
                except ValueError as e:
                    print(e)
            classes = list(classes)

            examples = cls._encode_output(examples, classes, output_type)

            train, test = cls._standard_creator(examples, **kwargs)

            if equalize_classes:
                while not (cls._classes_equal_perc(train) and cls._classes_equal_perc(test)):
                    train, test = cls._standard_creator(examples, **kwargs)

            return train, test, classes

