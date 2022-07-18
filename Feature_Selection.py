import pandas as pd
import DalexDatasets


class Feature_Selection():

    def __init__(self,dsConfig,config):
        self.config = config
        self.dsCofig = dsConfig

    def XAI_feature_selection(self,x_train,x_test,y_train,y_test):

        Dalex_on_Train, Dalex_on_Adversarial = DalexDatasets.DalexDatasets.loadDalexDatasets(self)

        Dalex_on_Train.rename(columns = {'variable':'Train', 'dropout_loss':'Train_ranking'}, inplace = True)
        Dalex_on_Adversarial.rename(columns = {'variable':'Adversarial', 'dropout_loss':'Adversarial_ranking'}, inplace = True)



        features_step = int(self.config.get('features_step'))

        Dalex_on_Train = Dalex_on_Train[ : features_step]
        Dalex_on_Adversarial = Dalex_on_Adversarial[ : features_step]

        for i in range(features_step) :
            Dalex_on_Train[[i,'Train_ranking']] = i
            Dalex_on_Adversarial[[i,'Adversarial_ranking']] = i


        l1 = Dalex_on_Train
        l2 = Dalex_on_Adversarial

        yu = []
        for i in range(l1.shape[0]):
            if l1.iloc[i]['Train'] in l2['Adversarial'].values:
                index = l1.iloc[i]['Train']
                dfb = int(l2[l2['Adversarial'] == index].index[0])

                yu.append((l1.iloc[i]['Train'], l1.iloc[i]['Train_ranking'], l2.iloc[dfb]['Adversarial_ranking']))

            else:
                yu.append((l1.iloc[i]['Train'], l1.iloc[i]['Train_ranking'], 999))
                print('')
            if l2.iloc[i]['Adversarial'] in l1['Train'].values:
                continue
            else:
                yu.append((l2.iloc[i]['Adversarial'], 999, l2.iloc[i]['Adversarial_ranking']))


        yu = pd.DataFrame(yu)

        features = []
        for i in range(yu.shape[0]):
            if yu.iloc[i][1] == 999 or yu.iloc[i][2] == 999:
                continue
            else:
                features.append(yu.iloc[i][0])

        #features.append(self.dsConfig.get('label'))
        print(features)
        XAIFS_train = x_train[features]
        XAIFS_test = x_test[features]
        _,_,XAIFS_adv,_ = DalexDatasets.DalexDatasets.read_T_A_Datasets(self,x_train,y_train)
        XAIFS_adv = XAIFS_adv[features]

        return XAIFS_train,XAIFS_adv,XAIFS_test


