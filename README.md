# ISMIS


### The repository contains code refered to the work:

# XAI to explore robustness of features in adversarial training for cybersecurity
Malik AL-Essa, Giuseppina Andresini, Annalisa Appice, Donato Malerba

# Cite this paper
AL-Essa, M., Andresini, G., Appice, A., Malerba, D. (2022). XAI to Explore Robustness of Features in Adversarial Training for Cybersecurity. In: Ceci, M., Flesca, S., Masciari, E., Manco, G., Raś, Z.W. (eds) Foundations of Intelligent Systems. ISMIS 2022. Lecture Notes in Computer Science(), vol 13515. Springer, Cham. https://doi.org/10.1007/978-3-031-16564-1_12


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
Two different types of datasets are used in this work CICICD17, and CIC-Maldroid20. MinMax scaler has been used to normalize the datasets. The datasets and models that have been used in work can be downloaded through [Datasets and Models](https://drive.google.com/drive/folders/1D60-5h4Bp4RC_P4qMHkCMYm6tWrc34tR)
  
  
   

### How to use

The implementation for all the experiments used in this work are listed in this repository.
  * main.py : to run ISMIS
 


## Replicate the Experiments

To replicate the experiments of this work, the models and datasets that have been saved in [Datasets and Models]( https://drive.google.com/drive/folders/1D60-5h4Bp4RC_P4qMHkCMYm6tWrc34tR)  can be used. Global Variable are saved in Conf.conf :

* ###### TRAIN_BASELINE = 0   &emsp;        #1 train baseline with hyperopt <br />
* ###### CREATE_ADVERSARIAL_SET=0 &emsp;  #if 1 create the adversarial samples <br />
* ###### Attack_Type =1      &emsp;  ## 1 for FGSM, 2 for BIM, and 3 for PGD <br />

* ###### train_Attack = 1             &emsp;      #0 not to train, 1 to train / To Train a model using adversarial training <br />
* ###### Dalex_relevance = 0   &emsp; # # 1 to compute dalex relevance, to compute the global feature relevance for the model <br />
* ###### Dalex_model= 6             &emsp;  #if 2 for baseline(V1), 6 for T_A model (V2) <br />
* ###### Dalex_Dataset_type = 0 &emsp;    # 1 for original dataset, 0 adversarial samples <br />
 
* ###### Dalex_train_dataset = 1      &emsp;          #1 for training dataset, 2 for Adversarial Dataset <br />

* ###### features_step = 15      &emsp;          #the number of features to be selected based on the intersection between XAI-Training and XAI_Adversarial <br />
* ###### XAIFS = 1      &emsp;          ## 1 to run XAIFS , XAIFS is the training of the model based on feature selection using XAI <br />

