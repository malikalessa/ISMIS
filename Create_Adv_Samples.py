import tensorflow as tf
import numpy as np
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent
import Global_Config as gc
from keras.models import load_model
from sklearn.utils import shuffle
import pandas as pd

class Create_Adv_Samples():

    def __init__(self, dsConfig, config):
        self.config = config
        self.ds = dsConfig

    ####### Create FastGradientMethod ##########

    def Adv_Samples(self,x_train,x_test, y_train, y_test):

        columns = x_train.columns
        x_train = np.asarray(x_train)
        path = self.ds.get('pathModels')
        model = load_model(path +self.ds.get('baseline_model') )
        n_class = self.ds.get('n_class')


        classifier = TensorFlowV2Classifier(model, nb_classes=int(self.ds.get('n_class')),input_shape=(1,x_train.shape[1]),
                                        loss_object = tf.keras.losses.CategoricalCrossentropy())  #, clip_values=(0,1))

        eps = float(self.ds.get('epsilon'))


        ##### Create FGSM Samples #######3
        if (int(self.config.get('Attack_Type'))==1) :
            attack = FastGradientMethod(estimator=classifier, eps=eps)
            adversarial_samples = attack.generate(x=x_train)

        ##### Create BIM Samples
        if (int(self.config.get('Attack_Type'))==2) :
            BIM = BasicIterativeMethod(estimator=classifier, eps=eps, eps_step=eps / 10, max_iter=10)
            adversarial_samples = BIM.generate(x=x_train)

        ########### Create PGD Samples ########
        if (int(self.config.get('Attack_Type'))==3):
            PGD = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=eps, eps_step=eps / 10, max_iter=10)
            adversarial_samples = PGD.generate(x=x_train)

        adversarial_original_samples = np.append(x_train, adversarial_samples, axis=0)
        y_label_adversarial = np.append(y_train, y_train, axis=0)


        adversarial_label = pd.DataFrame(y_train, columns = self.ds.get('attack'))
        adversarial_dataset = pd.DataFrame(adversarial_samples, columns =columns)

        adversarial_dataset = pd.concat([adversarial_dataset, adversarial_label], axis =1)
        adversarial_path = (self.ds.get('Adv_dataset')) + 'Adv_DS_' + \
                              (self.ds.get('Dataset_name')) + '_Attack_type_'+ (self.config.get('Attack_Type'))

        adversarial_dataset.to_csv(path_or_buf=adversarial_path +'.csv', index=False)


#        return adversarial_original_samples, y_label_adversarial, adversarial_samples


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
