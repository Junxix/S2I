## Towards Effective Utilization of Mixed-Quality Demonstrations in Robotic Manipulation via Segment-Level Selection and Optimization

<img src="./img/model.png" alt="xx picture" width="600">

### Run
For the data augmentation stage, run the command ` python ./augmentation/data_augmentation.py`  to execute the data augmentation script, which will preprocess and enhance the dataset.

Here are the argument explanations in the data augmentation process:
* `--dataset` : Specify the entire dataset used for the representation model training.
* `--video_path_ori` : The path where the visualizations of the augmented results will be stored. 
* `--total_images` ：The total number of images to generate during the augmentation process.
* `--numbers` ：The index or specific identifier used for data augmentation within the dataset. 

For the representation model training stage, run the command `bash command_train.sh`