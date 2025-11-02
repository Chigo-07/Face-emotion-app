Python 3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import tensorflow_datasets as tfds
... 
... print("Loading FER2013 dataset (this might take a few minutes the first time)...")
... ds_train, ds_info = tfds.load("fer2013", split="train", with_info=True)
... ds_test = tfds.load("fer2013", split="test")
... 
... print(ds_info)
... print("Example features from one sample:")
... for example in tfds.as_numpy(ds_train.take(1)):
...     print("Image shape:", example["image"].shape)
...     print("Label:", example["label"])
... 
SyntaxError: multiple statements found while compiling a single statement
