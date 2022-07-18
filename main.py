import sys
import configparser
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def readDataset(dsConfiguration):
    path=dsConfiguration.get('pathDataset')
    train=pd.read_csv(path+dsConfiguration.get('nameTrain'))
    test=pd.read_csv(path + dsConfiguration.get('nametest'))
    cls=dsConfiguration.get('label')
    y_train = train[cls]
    y_test = test[cls]

    try:
        train.drop([cls], axis=1, inplace=True)
        test.drop([cls], axis=1, inplace=True)

    except IOError:
        print('IOERROR')

    print(train.shape)

    return train,test, y_train,y_test


def main():

    dataset = sys.argv[1]
    config = configparser.ConfigParser()
    config.read('Conf.conf')

    #this contains path dataset and models
    dsConf = config[dataset]
    configuration = config['setting']
    pd.set_option('display.expand_frame_repr', False)

    x_train, x_test, y_train, y_test = readDataset(dsConf)
    print('tt',x_train.shape)



    if (int(configuration.get('TRAIN_BASELINE'))) :
       #### Creating Configurations 2
        import Baseline_Configurations as Baseline
        execution=Baseline.Baseline_Configurations(dsConf,configuration)
        execution.Baseline_model(x_train,x_test,y_train,y_test)

    if (int(configuration.get('Create_Adv_Samples'))):
        #### Creating Configurations 2
        import Create_Adv_Samples as Adv
        execution = Adv.Create_Adv_Samples(dsConf, configuration)
        execution.Adv_Samples(x_train, x_test, y_train, y_test)

    if (int(configuration.get('train_Attack'))) :
        import Baseline_Configurations as Baseline
        execution = Baseline.Baseline_Configurations(dsConf,configuration)
        execution.Model_trained_on_Adv_Samples(x_train,x_test,y_train,y_test)


    if (int(configuration.get('Dalex_relevance'))) :
        import DalexDatasets as dalex
        execution = dalex.DalexDatasets(dsConf,configuration)
        execution.createDalexDataset(x_train,x_test,y_train, y_test)

    if (int(configuration.get('XAIFS'))):
        import Baseline_Configurations as Baseline
        execution = Baseline.Baseline_Configurations(dsConf, configuration)
        execution.train_model_XAIFS(x_train,x_test,y_train,y_test)


if __name__ == "__main__":
    main()