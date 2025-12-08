import tensorflow_datasets as tfds
import tensorflow as tf
import os
import shutil

dataset_name = 'tf_flowers'
output_dir = 'data/raw'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

(ds, ds_info) = tfds.load(dataset_name, split='train', with_info=True)
label_names = ds_info.features['label'].names

print("Downloading dataset:", dataset_name)
print("Classes:", label_names)

for example in tfds.as_numpy(ds):
    image = example['image']
    label = example['label']
    class_name = label_names[label]

    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    img_count = len(os.listdir(class_dir))
    out_path = os.path.join(class_dir, f"{class_name}_{img_count}.jpg")
    tf.keras.preprocessing.image.save_img(out_path, image)

print("Dataset saved to:", output_dir)

