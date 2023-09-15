import os
import pathlib


working_path = pathlib.Path.cwd()
print("Current Path is",working_path)

folders = ('train', 'valid')
labels = ('0', '1')

input_path = pathlib.Path("Flood-Detector\Raw Data")
train_image_paths = sorted(input_path.rglob('train/*.png'))
valid_image_paths = sorted(input_path.rglob('test/*.png'))
print("Number of train images {} \nNumber of validation images {}".format(len(train_image_paths), len(valid_image_paths)))

#Creating the necessary directories.

for folder in folders:
    if not (working_path/folder).exists():
        (working_path/folder).mkdir()
    for label in labels:
        if not (working_path/folder/label).exists():
            (working_path/folder/label).mkdir()


try:
    for image_path in train_image_paths:
        if '_1' in image_path.stem:
            with (working_path/'train'/'1'/image_path.name).open(mode='xb') as f:
                f.write(image_path.read_bytes())
        else:
            with (working_path/'train'/'0'/image_path.name).open(mode='xb') as f:
                f.write(image_path.read_bytes())

except FileExistsError:
    print("Training images have already been moved.")
else:
    print("Training images moved.")

try:
    for image_path in valid_image_paths:
        if '_1' in image_path.stem:
            with (working_path/'valid'/'1'/image_path.name).open(mode='xb') as f:
                f.write(image_path.read_bytes())
        else:
            with (working_path/'valid'/'0'/image_path.name).open(mode='xb') as f:
                f.write(image_path.read_bytes())
except FileExistsError:
    print("Testing images have already been moved.")
else:
    print("Testing images moved.")