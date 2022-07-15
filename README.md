# ISMIS


### The repository contains code refered to the work:

# XAI to explore robustness of features in adversarial training for cybersecurity
Malik AL-Essa, Giuseppina Andresini, Annalisa Appice, Donato Malerba


![image](https://user-images.githubusercontent.com/38468857/178991622-3582906a-f8da-431b-9cd5-abc98e113c5d.png)





### Code Requirements

 * [Python 3.9](https://www.python.org/downloads/release/python-390/)
 * [Keras 2.7](https://github.com/keras-team/keras)
 * [Tensorflow 2.7](https://www.tensorflow.org/)
 * [Scikit learn](https://scikit-learn.org/stable/)
 * [Matplotlib 3.5](https://matplotlib.org/)
 * [Pandas 1.3.5](https://pandas.pydata.org/)
 * [Numpy 1.19.3](https://numpy.org/)
 * [Dalex 1.4.1](https://github.com/ModelOriented/DALEX)
 * [adversarial-robustness-toolbox 1.9](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
 * [scikit-learn-extra 0.2.0](https://scikit-learn-extra.readthedocs.io/en/stable/)
 * [Hyperopt 0.2.5](https://pypi.org/project/hyperopt/)


###  Description for this repository
Two different types of datasets are used in this work CICICD17, and CIC-Maldroid20. MinMax scaler has been used to normalize the datasets. The datasets and models that have been used in work can be downloaded through
* Datasets and Models.
  
  
   

### How to use

The implementation for all the experiments used in this work are listed in this repository.
  * main.py : to run ISMIS
 


## Replicate the Experiments

To replicate the experiments of this work, the models and datasets that have been saved in [Datasets and Models]  can be used. Global Variable are saved in Conf.conf :

* ###### TRAIN_BASELINE = 0   &emsp;        #1 train baseline with hyperopt <br />
* ###### CREATE_ADVERSARIAL_SET=0 &emsp;  #if 1 create the adversarial samples <br />
* ###### Attack_Type =1      &emsp;  ## 1 for FGSM  <br />

* ###### train_Attack = 1             &emsp;      #0 not to train, 1 to train / To Train a model using adversarial training <br />
* ###### local_shap_values = 1  &emsp; #if 1 to compute local shap values, 0 to load the saved values <br />
* ###### Config_model= 6             &emsp;  #if 2 for baseline(V1), 6 for T_A model (V2) <br />
* ###### Fine_Tuning = 0 &emsp;    #if 1 To fine-tune V2 (Adversarial training model). The model will be fine-tuned twice, using XAI and T+A <br />
 
* ###### Fine_Tuning_baseline = 0      &emsp;          ## 1 To fine tune the baseline(V1) model, The model will be fine tuned twice, using XAI and T <br />


