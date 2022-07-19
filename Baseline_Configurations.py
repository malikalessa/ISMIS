import Feature_Selection
import Global_Config
import report
from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import Baseline_HyperModel
import os
import Global_Config as gc
import Create_Adv_Samples
import pandas as pd


class Baseline_Configurations():

    def __init__(self, dsConfig, config):
        self.config = config
        self.dsConfig = dsConfig

    ##### Creating Baseline Model

    def Baseline_model(self, x_train,x_test, y_train, y_test):
        ##Hyperopt on T1 to learn  DNN####
        ########################### Config 2 #########################3
        path = self.dsConfig.get('pathModels')
        n_class = int(self.dsConfig.get('n_class'))
        gc.n_class = n_class
        report_name = path + 'Hyperopt_Config2.txt'
        try:
            os.remove(report_name)
        except:
            print('')

        config_No = 2

        model_hyperopt, time1,score = Baseline_HyperModel.hypersearch(x_train,y_train,x_test,y_test,path,config_No)
        model_hyperopt.save(path+ self.dsConfig.get('baseline_model'))
        # model_hyperopt = load_model(path + self.ds.get('baseline_model'))

        Y_predicted = np.argmax(model_hyperopt.predict(x_test), axis=1)
        Confusion_matrix = confusion_matrix(y_test, Y_predicted)
        Classification_report = classification_report(y_test, Y_predicted)
        Accuracy = accuracy_score(y_test, Y_predicted)
        print('Accuracy : ', Accuracy)
        name = 'Hyperopt_Configuration2'

        report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)
        return model_hyperopt

    def Model_trained_on_Adv_Samples(self,x_train,x_test, y_train,y_test):


        path = self.dsConfig.get('pathModels')
        n_class = int(self.dsConfig.get('n_class'))
        gc.n_class = n_class

            #### Adversarial Samples ##############################################################

        adversarial_original_samples, y_label_adversarial, adversarial_samples, _ = Create_Adv_Samples.Create_Adv_Samples. \
            read_T_A_Datasets(self, x_train, y_train)
        print('Adv Samples.shape', adversarial_samples.shape)

         ################ Config 6 #########################################3
        if (int(self.config.get('Dalex_model')) == 6):
            print('Training Conf-6')

            report_name = path + 'Hyperopt_Config_6.txt'

            ### Calling function to predict Adversarial Samples only

            print('Adv.shape used in conf-6', adversarial_original_samples.shape)
            print('Adv label shape used in conf-6 ', y_label_adversarial.shape)
            print('Classes : ', y_label_adversarial.value_counts())

            config_No = 6
            gc.Attack_Type = self.config.get('Attack_Type')
            model, time1, best_score = Baseline_HyperModel.hypersearch(adversarial_original_samples,
                                                                           y_label_adversarial,
                                                                           x_test, y_test, path, config_No)

            Y_predicted = np.argmax(model.predict(x_test), axis=1)
            Confusion_matrix = confusion_matrix(y_test, Y_predicted)
            Classification_report = classification_report(y_test, Y_predicted)
            Accuracy = accuracy_score(y_test, Y_predicted)
            print('Accuracy : ', Accuracy)

            try:
                 os.remove(report_name)
            except:
                print('')
            name = 'AdvSample_OriginalData'
            report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)


    def train_model_XAIFS(self, x_train, x_test, y_train,y_test):

        path = self.dsConfig.get('pathModels')
        n_class = int(self.dsConfig.get('n_class'))
        gc.n_class = n_class

        XAIFS_train,XAIFS_adv,XAIFS_test = Feature_Selection.Feature_Selection.XAI_feature_selection(self,x_train,x_test,y_train,y_test)
        print('Training dataset for XAIFS shape : ', XAIFS_train.shape)
        print('Testing dataset for XAIFS shape : ', XAIFS_test.shape)
        print('Adv dataset for XAIFS shape : ', XAIFS_adv.shape)

        XAIFS_train_adv = XAIFS_train.append(XAIFS_adv)
        y_train_adv = y_train.append(y_train)

        config_No = int(self.config.get('features_step'))

        print ('The Feature Selection has been done using ', str(config_No) + ' Features\n')

        gc.XAIFS = 'XAIFS'

        model, time1, best_score = Baseline_HyperModel.hypersearch(XAIFS_train_adv, y_train_adv,
                                                                   XAIFS_test, y_test, path, config_No)

        model.save(path + 'XAIFS_'+ str(config_No) + '.h5')

        Y_predicted = np.argmax(model.predict(XAIFS_test), axis=1)
        Confusion_matrix = confusion_matrix(y_test, Y_predicted)
        Classification_report = classification_report(y_test, Y_predicted)
        Accuracy = accuracy_score(y_test, Y_predicted)
        print('Accuracy : ', Accuracy)

        report_name = path +'XAIFS_' + str(config_No)+'.txt'
        try:
            os.remove(report_name)
        except:
            print('')
        name = 'XAIFS_'+str(config_No)
        report.report(report_name, Accuracy, Confusion_matrix, Classification_report, name)
