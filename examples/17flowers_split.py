import shutil
import os

directory = "jpg"
target_train = "17flowers"

if not os.path.isdir(target_train):
    os.makedirs(target_train)

classes = [
    "daffodil",
    "snowdrop",
    "lilyvalley",
    "bluebell",
    "crocus",
    "iris",
    "tigerlily",
    "tulip",
    "fritillary",
    "sunflower",
    "daisy",
    "coltsfoot",
    "dandelion",
    "cowslip",
    "buttercup",
    "windflower",
    "pansy",
]

j = 0
for i in range(1, 1361):
    label_dir = os.path.join(target_train, classes[j])

    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)

    filename = "image_" + str(i).zfill(4) + ".jpg"
    shutil.copy(
        os.path.join(directory, filename), os.path.join(label_dir, filename)
    )

    if i % 80 == 0:
        j += 1