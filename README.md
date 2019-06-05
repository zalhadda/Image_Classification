### train.py

To train the network, call the main() function. As of now, the main() function is not called.

The program can be run with: python train.py --data <images_directory> --save <model_saved_directory>

The default <images_directory> is "tiny_imagenet/"
The default <model_saved_directory> is "trained_model/" and the default saved model name is "model.dat"

So, you could run "python train.py" as long as you have a "tiny_imagenet" folder with the train & val images.

The train and val images have to be in separate "train" and "val" folders.

"Val" images are expected to be in a sub-folder called "images". So, "tiny_imagenet/val/images"
"Train" images are expected to be in "tiny_imagenet/train/"

The train images are assumed to already be organised into subcategorial folders.

If the val images are not organised into subcategorical folders, the program will organise them.

The program was trained over four epochs to achieve a 51.1% validation accuracy. Training it for four epochs took 8-9 hours on my computer, otherwise I would have done more epochs.


### test.py

I didn't have a working webcam. There is code in the test.py for accessing a webcam, but I haven't been able to test if it works. Therefore, I've also included code to access random test images and display them through matplotlib.

To switch between the webcam and random test images, set TEST_CAM to 1 (for webcam) or 0 (for random images). The TEST_CAM variable is in the code.

The program can be run with: python test.py --model <saved_model_directory>

The default saved_model_directory is "trained_model/" and it is assumed the model will be called "model.dat"

For testing with the random images, it is expected that the images will be in "tiny_imagenet/test/images"

For labeling images, it is expected there will be a "words.txt" file in "tiny_imagenet/" with the categorisation_id and the accompanying label(s).

Make sure main() is not called in train.py before running test.py, otherwise it will train the entire model.