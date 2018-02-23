# SVHN-Classifier
Simple classifier to classify SVHN images, based on Keras with the Tensorflow backend.

### Requirements:
* Keras 2.1.4
* Numpy 1.14.1

### To predict images:
To predict existing images with the pre-trained model (95.45% accuracy on the SVHN test set)
* `python svhn_classifier.py --predict --model weights.hdf5 --img_path path-to-images`

Images should be stored in the following layout:
path-to-images
    * class-0
        * img1.jpg
        * img2.jpg
        * ...
    * class-1
        * img1.jpg
        * img2.jpg
        * ...
    * ...


### To train a new classifier
Download the SVHN data set:
* go to http://ufldl.stanford.edu/housenumbers/
* download the cropped digits (Format 2): train_32x32.mat and test_32x32.mat
* run `python preprocess_svhn.py --data path-to-the-downloaded-files --save_to where-to-save-normalized-data`

To train a new classifier on the SVHN data:
* `python svhn_classifier.py --train`

To view training statistics:
* `tensorboard --logdir log_dir/`

Check out command line arguments for further control over the hyperparameters used for training:
* `python svhn_classifier.py --help`

