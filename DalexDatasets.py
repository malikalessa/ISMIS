import pandas as pd
import os
import dalex as dx
from keras.models import load_model
from sklearn.utils import shuffle
from math import sqrt
from sklearn import preprocessing


import numpy as np


class DalexDatasets():
    def __init__(self, dsConfigConfig,config):
        self.config = config
        self.dsConfig = config

    def read_T_A_Datasets(self, train_dataset, y_train):

        path = (self.dsConfig.get('Adv_dataset')) + 'Adv_DS_' + \
                           (self.dsConfig.get('Dataset_name')) + '_Attack_type_'+ (self.config.get('Attack_Type'))

        train_adv_samples = pd.read_csv(path +'.csv')
        cls = self.dsConfig.get('label')
        y_train_adv = train_adv_samples[cls]

        try:
            train_adv_samples.drop([cls], axis=1, inplace=True)
        except IOError:
            print('IOERROR')
        print('Train Adv Dataset shape : ', train_adv_samples.shape)
        print('YTrain Adv Dataset shape : ', y_train_adv.shape)

        train_dataset = train_dataset.append(train_adv_samples)
        y_train = y_train.append(y_train_adv)

        print('\nTrain Dataset+Adversarial Samples shape : ', train_dataset.shape)
        print('\nYTrain Dataset+Adversarial Samples shape : ', y_train.shape)

        return train_dataset, y_train, train_adv_samples, y_train_adv


    def createDalexDataset(self, x_train,x_test,y_train,y_test):
        # To compute the feature relevance in the models

        model = load_model(self.dsConfig.get('pathModels') + 'ConfigNo6Attack_Type_' + self.config.get('Attack_Type') + '_NN.h5')
        config = int(self.config.get('Dalex_model'))


        if (int(self.config.get('Dalex_Dataset_type')) == 1):
            x_train,y_train = x_train,y_train
            print('The Original Dataset is used in Dalex')


        if (int(self.config.get('Dalex_Dataset_type')) == 0):
            print(' Adversarial Samples are used in Dalex')
            x_train,y_train, adv_samples, y_train_adv = self.read_T_A_Datasets(x_train,y_train)

            x_train.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)


        print('x_train....', x_train.shape)





        if (int(self.config.get('Dalex_train_dataset')) == 1):

            print( 'Train Dataset is used in Dalex')
            dataset_path = self.dsConfig.get('pathDalexDataset') + self.dsConfig.get('Dataset_name') + '_Dalex_on_Train.csv'
            train_dataset = x_train
            y_labels = y_train


        if (int(self.config.get('Dalex_train_dataset')) == 2):
            print('Adversarial Dataset is used in Dalex')
            dataset_path = self.dsConfig.get('pathDalexDataset') + self.dsConfig.get('Dataset_name') + 'Dalex_on_Adversarial_Samples.csv'
            train_dataset = adv_samples
            y_labels = y_train_adv

        print('\nTrain Dataset : ', train_dataset.shape)
        print('\nYTrain Dataset : ', y_labels.value_counts())


        explainer = dx.Explainer(model, train_dataset, y_labels)
        explanation = explainer.model_parts(random_state=42)
        variable_importance = pd.DataFrame(explanation.result)

        variable_importance = variable_importance.sort_values(by=['dropout_loss'], ascending=False)

        variable_importance.drop(['label'], axis=1, inplace=True)

        variable_importance = variable_importance[variable_importance.variable != '_full_model_']
        variable_importance = variable_importance[variable_importance.variable != '_baseline_']


        print("Dalex File Shape  : ", variable_importance.shape)

        variable_importance.to_csv(path_or_buf=dataset_path, index=False)




    def loadDalexDatasets(self):


        attack_type = int(self.config.get('Attack_Type'))
        print(attack_type)
        if attack_type == 1:
            attack_path = 'FGSM/'
        elif attack_type == 2:
            attack_path = 'BIM/'
        elif attack_type == 3:
            attack_path = 'PGD/'

        path = self.dsConfig.get('pathDalexDataset') + attack_path + self.dsConfig.get('Dataset_name')

        Dalex_on_Train = pd.read_csv(path + '_Dalex_on_Train.csv')
        Dalex_on_Adversarial = pd.read_csv(path + '_Dalex_on_Adversarial_Samples.csv')

        print('The Dalex Files has been uploaded')

        return Dalex_on_Train, Dalex_on_Adversarial


