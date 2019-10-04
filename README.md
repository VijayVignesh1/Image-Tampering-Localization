# SPi-Assignment
Finding if an image is tampered and if tampered, the tampered section is localized.
1.Run "python data_process.py" initially to create two h5py files (This is for classification process), one for training and the other for validation.
  In data_process.py, I take 12000 cat images, tamper it manually. The process of Dodging is used for manipulation, though it can be extended to other processes as well. Different parts of the images are tampered. The size of the training data becomes 24000.
  Then, I save both the tampered and Normal image, along with the appropriate labels in a h5py file. This is repeated for validation dataset.
2. Run "python data_process_unet.py" initially to create two h5py files (This is for localization process), one for training and the other for validation.
   Here, 1000 cat images are taken and tampered (dodging). The tampered images with the corresponding masks are stored in the h5py files.
3. Run "python train.py". This loads the dataset and trains the classification model and saves the checkpoint. I have already trained the model for a few epochs.
   Then run "python train_unet.py". This loads the dataset and trains the localization model (UNet) and saves the checkpoint. I have already trained the model for a few epochs.
   If you wish to train further, please change the "epochs" variable in train.py and train_unet.py and run it. Validation accuracy is also calculated using the validation data. Validation accuracy for the classification model is 90%.(500 Normal Images and 500 Manipulated Images)
   I have also added a test file to test any image. Run "python test.py" to test using default parameters. Run "python test.py --help" to see customizable parameters. As of now, checkpoint of classifer, checkpoint of localizer, image, tamper (True/False) and side of tampering are customizable.

