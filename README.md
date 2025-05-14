# REMI - Real-time Ipman-R Inference and Analysis

**REMI** is a system for real-time 3D human pose estimation and dynamic visualization, designed to provide immediate, intuitive feedback on movement quality. This repository contains the modified Ipman-R code, which forms the core of the real-time 3D human pose and shape estimation component of the REMI project.

**Main REMI Project Page:** (https://dominickstephens.github.io/REMI_page/)

**Aitviewer (Real-time Visualization) Component:** [https://github.com/Dominickstephens/aitviewer-skel-RT](https://github.com/Dominickstephens/aitviewer-skel-RT)

## Overview of this Repository (ipman-r-RT)

This repository focuses on the real-time inference capabilities of the Ipman-R model. It includes scripts to:
* Run live 3D pose and shape estimation from a webcam.
* Visualize the output locally.
* Send the estimated parameters to a remote visualization client (like the modified Aitviewer).
* Perform detailed performance and stability analysis of the inference process.

## Scripts
``` bash
python webcam.py
```
This script runs real-time Ipman-R inference on a live webcam feed. It captures video, performs 3D pose and shape estimation using the HMR model and SMPL parameters, and then renders the 3D mesh overlaid directly onto the webcam image in an OpenCV window. It includes FPS display. This is a good script for a quick local visualization of the model's output.

``` bash
python webcam_only.py
```
A basic utility script that captures and displays the raw webcam feed. It does not perform any pose estimation or 3D rendering. Useful for checking camera functionality.

``` bash
python webcam_metrics.py
```
This is a comprehensive script for robust performance evaluation of the real-time Ipman-R inference.
Key features:

- Runs real-time inference from the webcam.
- Metrics Calculation: Utilizes a separate thread to calculate detailed performance and stability metrics, including:
  - Processing FPS (Frames Per Second)
  - End-to-End Latency (ms)
  - Average Pose Change (Euclidean Distance between consecutive frames' rotation matrices)
  - Average Translation Change (mm)
  - Average Joint Position Change (mm, based on vertex positions)
  - Average Shape Parameter Variance
  - Average Detection/Tracking Rate (%)
- GUI Interface: Provides a Tkinter GUI to select the recording "Condition" (e.g., Optimal, Low Light) and trigger a 5-second recording of averaged metrics.
CSV Logging: Saves the averaged metrics to a CSV file (smpl_metrics_socket_threaded_avg.csv).
- Socket Communication: Attempts to connect to a relay server (localhost:9999) to send SMPL parameters, allowing for decoupled visualization if a server (like Aitviewer) is running.
- Local Display (Optional): Can display the webcam feed with the SMPL overlay and instantaneous FPS/Latency.
This script is crucial for quantitatively analyzing the system's performance and robustness under various conditions.

``` bash
python webcam_client.py
```

This script acts as a dedicated client for sending real-time Ipman-R inference results to a remote visualization server (e.g., Aitviewer via a relay).
Key features:

- Performs real-time 3D pose and shape estimation from the webcam.
- Pose Conversion: Converts the model's output rotation matrices to axis-angle representation.
- Orientation Adjustment: Applies a 180-degree rotation around the X-axis to the global orientation, a common adjustment for compatibility with different coordinate systems or renderers.
- Socket Communication: Connects to a relay server (localhost:9999) and sends the processed SMPL parameters (body pose, root orientation, shape betas, and translation) using a custom packet structure (pickle serialization with a magic number and CRC32 checksum).
- Local Display (Optional): Can also display the webcam feed with the SMPL overlay locally for debugging or direct viewing.

``` bash
python webcam_bicep.py
```
This script is the first attempt at utilisng the captured SMPL pose to drive nuanced excercise information such as counting excercieses (bicep curl) from a 3D dimensional POV.

## For Installation, follow the below instructions


# Intuitive Physics-Based Humans - Regression (IPMAN-R) [CVPR-2023] 

> Code repository for the paper:  
> [**3D Human Pose Estimation via Intuitive Physics**](https://arxiv.org/abs/2303.18246)  
> [Shashank Tripathi](https://sha2nkt.github.io), [Lea M&uuml;ller](https://ps.is.mpg.de/person/lmueller2), [Chun-Hao Paul Huang](https://ps.is.mpg.de/person/chuang2), [Omid Taheri](https://www.is.mpg.de/person/otaheri), [Michael J. Black](https://ps.is.tuebingen.mpg.de/person/black), [Dimitrios Tzionas](https://ps.is.tuebingen.mpg.de/person/dtzionas)  
> *IEEE Computer Vision and Pattern Recognition (CVPR), 2023*

[[Project Page](https://ipman.is.tue.mpg.de)] [[Paper](https://arxiv.org/abs/2303.18246)] [[Video](https://www.youtube.com/watch?v=Dufvp_O0ziU)] [[Poster](https://drive.google.com/file/d/1n8QeOI_WRqcVDUMrB-lG2NCJURhBjppG/view?usp=sharing)] [[Data (MoYo)](https://moyo.is.tue.mpg.de/)] [[License](https://ipman.is.tue.mpg.de/license.html)] [[Contact](mailto:ipman@tue.mpg.de)]

![teaser](assets/main_teaser_cropped.gif)

## Installation instructions

IPMAN-R has been implemented and tested on Ubuntu 20.04 with python >= 3.7.

Clone the repository and install the requirements. 
```
git clone https://github.com/sha2nkt/ipman-r.git
cd ipman-r
conda create -n ipman_p37 python=3.7
pip install -U pip
pip install torch==1.1.0 torchvision==0.3.0
pip install neural-renderer-pytorch
pip install -r requirements.txt
```

After finishing with the installation, you can continue with running the demo/evaluation/training code.
In case you want to evaluate our approach on Human3.6M, you also need to manually install the [pycdf package of the spacepy library](https://pythonhosted.org/SpacePy/pycdf.html) to process some of the original files. If you face difficulties with the installation, you can find more elaborate instructions [here](https://stackoverflow.com/questions/37232008/how-read-common-data-formatcdf-in-python).

## Fetch data
We provide a script to fetch the necessary data for training and evaluation. You need to run:
```
chmod +x fetch_data.sh
./fetch_data.sh
```
The GMM prior is trained and provided by the original [SMPLify work](http://smplify.is.tue.mpg.de/), while the implementation of the GMM prior function follows the [SMPLify-X work](https://github.com/vchoutas/smplify-x). Please respect the license of the respective works.

Besides these files, you also need to download the *SMPL* model. You will need the [neutral model](http://smplify.is.tue.mpg.de) for training and running the demo code, while the [male and female models](http://smpl.is.tue.mpg.de) will be necessary for evaluation on the 3DPW dataset. Please go to the websites for the corresponding projects and register to get access to the downloads section. In case you need to convert the models to be compatible with python3, please follow the instructions [here](https://github.com/vchoutas/smplx/tree/master/tools).

Due to license restrictions, we are unable to release Human3.6M SMPL fits. If you want to train on Human3.6M, you need to download the [original dataset](http://vision.imar.ro/human3.6m/description.php) and run the [MoSh code](http://mosh.is.tue.mpg.de/) to obtain the SMPL parameters. Models trained without Human3.6M can still be used for evaluation, but will not be able to achieve the same performance.

## Run IPMAN-R evaluation
We provide code to evaluate our models on the datasets we employ for our empirical evaluation. We provide the required npzs. For details on how we obtain the npzs, please follow the [details for data preprocessing](datasets/preprocess/README.md).

Example usage:
```
python eval.py --checkpoint data/ipman_checkpoints/2022_03_01-06_31_55_epoch1_stepcount_24000.pt --dataset rich-test --log_freq=20 --vis_path 'dummy'
```

To also evaluate stability metrics, please add the `--eval_stability` flag.
```
python eval.py --checkpoint data/ipman_checkpoints/2022_03_01-06_31_55_epoch1_stepcount_24000.pt --dataset rich-test --log_freq=20 --vis_path 'dummy' --eval_stability
```


Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. RICH ```--dataset=rich-test```
2. Human3.6M Protocol 1 ```--dataset=h36m-p1```
3. Human3.6M Protocol 2 ```--dataset=h36m-p2```
4. 3DPW ```--dataset=3dpw```
5. LSP ```--dataset=lsp```
6. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```


You can also save the results (predicted SMPL parameters, camera and 3D pose) in a .npz file using ```--result=out.npz```.

For the MPI-INF-3DHP dataset specifically, we include evaluation code only for MPJPE (before and after alignment). If
you want to evaluate on all metrics reported in the paper you should use the official MATLAB test code provided with the
dataset together with the saved detections.

## Run training code
Due to license limitiations, we cannot provide the SMPL parameters for Human3.6M (recovered using [MoSh](http://mosh.is.tue.mpg.de)). Even if you do not have access to these parameters, you can still use our training code using data from the other datasets. Again, make sure that you follow the [details for data preprocessing](datasets/preprocess/README.md).

Example usage:
```
python train.py --cfg config.yaml 
```
You can view the full list of command line options in config.yaml. The default values are the ones used to train the models in the paper. We used Human3.6M, LSP, MPI-INF-3DHP, LSPET, COCO, MPII and RICH to train the final model. 

Running the above command will start the training process. It will also create the folders `logs` and `logs/train_example` that are used to save model checkpoints and Tensorboard logs.
If you start a Tensborboard instance pointing at the directory `logs` you should be able to look at the logs stored during training.

## Citing
If you find this code useful for your research or the use data generated by our method, please consider citing the following paper:

```bibtex
@inproceedings{tripathi2023ipman,
    title = {{3D} Human Pose Estimation via Intuitive Physics},
    author = {Tripathi, Shashank and M{\"u}ller, Lea and Huang, Chun-Hao P. and Taheri Omid
    and Black, Michael J. and Tzionas, Dimitrios},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition (CVPR)},
    month = {June},
    year = {2023}
}
```

## License

See [LICENSE](LICENSE).

## Acknowledgments

This code is heavily derived from [SPIN](https://github.com/nkolot/SPIN) from Kolotouros et al. We also thank Tsvetelina Alexiadis, Taylor McConnell, Claudia Gallatz, Markus Höschle, Senya Polikovsky, Camilo Mendoza, Yasemin Fincan, Leyre Sanchez and Matvey Safroshkin for data collection, Giorgio Becherini for MoSh++, Joachim Tesch and Nikos Athanasiou for visualizations, Zincong Fang, Vasselis Choutas and all of Perceiving Systems for fruitful discussions. This work was funded by the International Max Planck Research School for Intelligent Systems (IMPRS-IS) and in part by the German Federal Ministry of Education and Research (BMBF), Tübingen AI Center, FKZ: 01IS18039B.

## Contact

For technical questions, please create an issue. For other questions, please contact `ipman@tue.mpg.de`.

For commercial licensing, please contact `ps-licensing@tue.mpg.de`.
