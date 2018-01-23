# SfMLearner Chainer version
This codebase implements the system described in the paper:

Unsupervised Learning of Depth and Ego-Motion from Video [link](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/)  
See the [project webpage](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/) for more details.

TF code: https://github.com/tinghuiz/SfMLearner

初心者になんか参考になりそうなページ: https://qiita.com/peisuke/items/fbe00bacb22df8115323

## TODO
- test, evaluator

## Preparing training data
In order to train the model using the provided code, the data needs to be formatted in a certain manner.

For [KiTTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command
```bash
python data/prepare_train_data.py /path/to/KITTI_raw --dataset-format kitti --static-frames ./data/static_frames.txt  --dump-root /path/to/KITTI_formatted --height 128 --width 416 --num-threads 8
```

## Training using KiTTI Raw Dataset
Once the data are formatted following the above instructions, you should be able to train the model by running the following command
```bash
python3 train.py experiments/sfm_learner_v1.yml
```


## Testing using KiTTI Raw Dataset
Once you have model trained, you can obtain the single-view depth predictions on the KITTI eigen test split formatted properly for evaluation by running
```bash
python test_kitti_depth.py --dataset_dir /path/to/raw/kitti/dataset/ --output_dir /path/to/output/directory --ckpt_file /path/to/pre-trained/model/file/
```
Again, a sample model can be downloaded by
```bash
bash ./models/download_depth_model.sh
```


## トレーニングする際の注意点
- とりあえず、kittiのraw datasetを用いてトレーニングを行います。
- フレーム数: Depthは3, Poseは5を利用。中心画像をTargetとして利用
- KITTIデータでは、左右両方の画像を独立なデータとして利用
- Optical Flowを用いて動きが見られないデータを削除(おそらくstatic_frames.txtがそのリスト)

## IDEA
- Explanation Maskを正確に学習できると、その結果をsegmentationに応用できそう
- 同じく、Edge Detection, depth推定も精度が向上しそう
