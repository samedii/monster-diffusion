from datastream import Datastream, samplers

from .datasets import datasets


def evaluate_datastreams():
    datasets_ = datasets()
    for split_name, dataset in datasets_.items():
        print(split_name, len(dataset))

    # def augment(example):
    #     crop.seed_(abs(example.hash()))
    #     pad.seed_(abs(example.hash()))
    #     return example.augment(crop).augment(pad)

    return {
        split_name: Datastream(
            dataset,
            samplers.SequentialSampler(len(dataset)),
        )
        for split_name, dataset in datasets_.items()
    }
