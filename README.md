# SPi-Assignment
Finding if an image is tampered.
1.Run "python data_process.py" initially to create two h5py files, one for training and the other for validation.
  In data_process.py, I take 12000 cat images, tamper it manually, by drawing a random line in the image. So, the size of the training data becomes 24000.
  Then, I save both the tampered and Normal image, along with the appropriate labels in a h5py file. This is repeated for validation dataset.
  Finally, two files are created. "CAT_IMAGES.h5py" and "CAT_IMAGES_VAL.h5py".
2. Then run "python train.py". This loads the dataset and trains a model and saves the checkpoint. I have already trained the model for two epochs.
   If you wish to train further, please change the "epochs" variable in train.py and run it. Validation accuracy is also calculated using the validation
   data.(500 Normal Images and 500 Manipulated Images)
   I have also added a test function to test any image from the validation set and see the output.

