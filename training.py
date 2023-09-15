from fastai.vision import *
from fastai.data.all import *
from fastai.vision.all import *
import pathlib

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

    working_path = pathlib.Path.cwd()

    dataloaders = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = GrandparentSplitter(),
    get_y = parent_label,
    item_tfms = [Resize(192, method='squish')]
    ).dataloaders(working_path, bs=32)

    learner = vision_learner(dataloaders, resnet18, metrics=error_rate)
    learner.fine_tune(9)

    augmented_dataloaders = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = GrandparentSplitter(),
    get_y = parent_label,
    item_tfms = RandomResizedCrop(192, min_scale=0.5),
    batch_tfms=aug_transforms()
    ).dataloaders(working_path, bs=32)



if __name__ == '__main__':
    run()