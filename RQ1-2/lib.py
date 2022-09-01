import os
import statistics

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset,BankDataset,MEPSDataset19
import aif360.metrics
from sklearn.metrics import accuracy_score
import numpy.polynomial.polynomial as poly
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

import aif360.datasets
import aif360.algorithms.preprocessing
import aif360.algorithms.inprocessing
import aif360.algorithms.postprocessing
from IPython.display import Markdown, display
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import binary_label_dataset

def fit_string_data(dataset, headlist,protected_attribute,privileged_class_dic):
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder()
    priviledgedclasslist = []
    for each in headlist:
        dataset[each] = lb.fit_transform(dataset[each])

        if each in protected_attribute:
            lb_name_mapping = dict(zip(lb.classes_, lb.transform(lb.classes_)))
            priviledgedclass = lb_name_mapping[privileged_class_dic[each]]
            priviledgedclasslist.append([priviledgedclass])
    return dataset,priviledgedclasslist


def get_data(datasetname,protectedattribute):
    dataset_used = datasetname  # "adult", "german", "compas"
    protected_attribute_used = protectedattribute  # 1, 2

    privileged_groups = [{protectedattribute: 1}]
    unprivileged_groups = [{protectedattribute: 0}]

    if dataset_used == "adult":
        dataset_orig = AdultDataset()
        #dataset_orig = get_adult()

    elif dataset_used == "german":
        dataset_orig = GermanDataset()

    elif dataset_used == "compas":
        # dataset_orig = CompasDataset(categorical_features=[])
        dataset_orig = CompasDataset()

    elif dataset_used == 'bank':
        dataset_orig = BankDataset()
    elif dataset_used == 'meps':
        dataset_orig = MEPSDataset19()
    elif dataset_used == 'ricci':
        dataset_orig = getdata_ricci()
    elif dataset_used == 'credit':
        dataset_orig = getdata_credit()


    return dataset_orig,privileged_groups,unprivileged_groups

def get_adult():
    traindatafilepath = './dataset/adult.data.txt'

    names =  ['age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income-per-year']

    headlist = ['fnlwgt','workclass', 'education',
                     'marital-status', 'occupation', 'relationship',
                     'native-country','race','sex']

    # preprocess training data
    whole_datset_use = pd.read_csv(traindatafilepath, names=names)
    protected_attribute = ['race', 'sex']
    privileged_class_dic = {}
    privileged_class_dic['race'] = ' White'
    privileged_class_dic['sex'] = ' Male'
    whole_datset_use,priviledgedclasslist = fit_string_data(whole_datset_use, headlist,protected_attribute,privileged_class_dic)

    dataset_orig = aif360.datasets.StandardDataset(favorable_classes=['>50K', '>50K.'],
                                                          df=whole_datset_use,
                                                          label_name='income-per-year',
                                                   privileged_classes=priviledgedclasslist,
                                                          protected_attribute_names=protected_attribute)
    print(dataset_orig.favorable_label)

    return dataset_orig



def get_PV_classic(classicclassifier, X, y):
    '''
    Description
        This function calculates the PV (model fit) evaluation result of a classic classifier and a training data set.

    Parameters
        classifier: the learner adopted to train the data; type: sklearn classifier
        X: the training data instances (without labels); type: numpy.ndarray
        y: training data labels; type: numpy.ndarray

    Returns
        PV: the degree of fit between classifier and data; type: float

    Examples
        from sklearn.tree import DecisionTreeClassifier
        from sklearn import datasets
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        dt = DecisionTreeClassifier()
        pv = lib.get_PV_classic(dt,X,y)
    '''
    noisedegreelist = np.arange(0.0,0.31,0.1) # label noise sequence [0,0.1,0.2,0.3]
    label_list = list(set(list(y)))

    dic_label_num = {} # get the number of instances for each class

    for label in label_list:
        dic_label_num[label] = (y == label).sum()

    model = classicclassifier

    trainingacculist = [] # list to put perturbed training accuracy

    cnt = 0
    while (cnt < len(noisedegreelist)):
        noisedegree = noisedegreelist[cnt]
        Y_changed = np.copy(y)
        dic_label_newnum = {} # record the number of perturbed samples for each class

        for label in label_list:
            dic_label_newnum[label] = 0 #initialize the dic; nothing is perturbed at the beginning

        for i in range(0, Y_changed.size):
            cnnn = 0
            for label in label_list:
                # perturb the label only if the perturbed labels in this class are fewer than required,
                if Y_changed[i] == label_list[cnnn] and dic_label_newnum[label_list[cnnn]] < float(dic_label_num[label_list[cnnn]]) * noisedegree:
                    try:
                        # replace the current label with its right neighbour label in label_list
                        Y_changed[i] = label_list[cnnn+1]
                        dic_label_newnum[label_list[cnnn]] += 1
                    except:
                        # if the label is the last in the label_list, replace it with the first element in label_list
                        Y_changed[i] = label_list[0]
                        dic_label_newnum[label_list[cnnn]] += 1
                    continue
                cnnn += 1


        model.fit(X, Y_changed) # retrain the model with perturbed labels
        y_changed_predictions = model.predict(X)
        trainaccuracy_perturbed = accuracy_score(Y_changed, y_changed_predictions)
        trainingacculist.append(trainaccuracy_perturbed)
        cnt+=1

    Ytest = trainingacculist

    Xtest = noisedegreelist
    m, b = poly.polyfit(Xtest, Ytest, 1) # conduct linear regression; b is the coefficient
    pv = 1-abs(1-abs(b)) # mirror PV by one, so that PV increases up to 1, then worsen afterwards
    #pv = -b

    # print('Perturbation results:')
    # for i in np.arange(0, 4, 1):
    #     print('Label noise degree: ' + str(round(noisedegreelist[i], 2)) + '  Training accuracy: ' + str(
    #         round(trainingacculist[i],2)))
    #
    # print('PV: '+str(round(pv,2)))
    # print('----------')

    return pv


def getdata_ricci():
    datasetfilepath = './dataset/RicciData.csv'

    dataset = pd.read_csv(datasetfilepath)
    #dataset = dataset.drop(columns=['Date','Name','Victim Sex','Victim Race'])



    proc_dataset= dataset.replace(['W','O'],[1,0])

    protected_attribute = ['Race']

    proc_dataset = fit_string_data(proc_dataset,['Position'])

    dataset_orig_bin = aif360.datasets.BinaryLabelDataset(favorable_label=1,
                                                                unfavorable_label=0,
                                                                df=proc_dataset,
                                                                label_names=['Class'],
                                                                protected_attribute_names=protected_attribute)


    return dataset_orig_bin
def get_average(filepath):
    readfile = open(filepath)
    lines = readfile.readlines()
    newfilepath = filepath.replace('.csv','_average.csv')
    writefile = open(newfilepath,'w')
    writefile.write('trainsizeratio' + ','
                    + 'featurenum' + ','
                    + 'depth' + ','
                    + 'train.mean_difference' + ','
                    + 'testpred.accuracy' + ','
                    + 'testpred.recall' + ','
                    + 'testpred.precision' + ','
                    + 'testpred.f1' + ','
                    + 'testpred.false_alarm' + ','
                    + 'testpred.equal_opportunity_difference' + ','
                    + 'testpred.statistical_parity_difference' + ','
                    + 'testpred.average_abs_odds_difference' + ','
                    + 'testpred.disparate_impact' + ','  # 9
                    + '\n'  # 10
                    )
    dic_string_meandiff = {}
    dic_string_accuracy = {}
    dic_string_recall = {}
    dic_string_precision = {}
    dic_string_f1 = {}
    dic_string_false = {}
    dic_string_equopp = {}
    dic_string_statis = {}
    dic_string_averageodd = {}
    dic_string_disparate = {}

    for thisline in lines:

        if 'datasetname' in thisline:
            continue
        splits = thisline.split(',')
        featurenum = splits[3]
        if featurenum == '1' or featurenum == '2':
            continue
        threestring = splits[2]+','+splits[3]+','+splits[4]
        if threestring in dic_string_meandiff:
            dic_string_meandiff[threestring].append(abs(float(splits[5])))
            dic_string_accuracy[threestring].append(abs(float(splits[6])))
            dic_string_recall[threestring].append(abs(float(splits[7])))
            dic_string_precision[threestring].append(abs(float(splits[8])))
            dic_string_f1[threestring].append(abs(float(splits[9])))
            dic_string_false[threestring].append(abs(float(splits[10])))
            dic_string_equopp[threestring].append(abs(float(splits[11])))
            dic_string_statis[threestring].append(abs(float(splits[12])))
            dic_string_averageodd[threestring].append(abs(float(splits[13])))
            dic_string_disparate[threestring].append(abs(1 - float(splits[14])))
        else:
            dic_string_meandiff[threestring] = [abs(float(splits[5]))]
            dic_string_accuracy[threestring] = [abs(float(splits[6]))]
            dic_string_recall[threestring] = [abs(float(splits[7]))]
            dic_string_precision[threestring] = [abs(float(splits[8]))]
            dic_string_f1[threestring] = [abs(float(splits[9]))]
            dic_string_false[threestring] = [abs(float(splits[10]))]
            dic_string_equopp[threestring] = [abs(float(splits[11]))]
            dic_string_statis[threestring] = [abs(float(splits[12]))]
            dic_string_averageodd[threestring] = [abs(float(splits[13]))]
            dic_string_disparate[threestring] = [abs(1 - float(splits[14]))]

    my_dict = dic_string_disparate
    alllist = []
    for eachk in my_dict:
        alllist += my_dict[eachk]
    key_max = max(alllist)
    key_min = min(alllist)
    print((0.929-key_min)*1.0/(key_max-key_min))


    for eachkey in dic_string_disparate:
        thislist = []
        for eachelement in dic_string_disparate[eachkey]:
            newelement = (eachelement-key_min)*1.0/(key_max-key_min)
            thislist.append(newelement)
        dic_string_disparate[eachkey] = thislist

    for each in dic_string_meandiff:
        # print(each)
        writefile.write(each + ','
                        + str(1.0 * sum(dic_string_meandiff[each]) / len(dic_string_meandiff[each])) + ','
                        + str(1.0 * sum(dic_string_accuracy[each]) / len(dic_string_accuracy[each])) + ','
                        + str(1.0 * sum(dic_string_recall[each]) / len(dic_string_recall[each])) + ','
                        + str(1.0 * sum(dic_string_precision[each]) / len(dic_string_precision[each])) + ','
                        + str(1.0 * sum(dic_string_f1[each]) / len(dic_string_f1[each])) + ','
                        + str(1.0 * sum(dic_string_false[each]) / len(dic_string_false[each])) + ','
                        + str(1.0 * sum(dic_string_equopp[each]) / len(dic_string_equopp[each])) + ','
                        + str(1.0 * sum(dic_string_statis[each]) / len(dic_string_statis[each])) + ','
                        + str(1.0 * sum(dic_string_averageodd[each]) / len(dic_string_averageodd[each])) + ','
                        + str(1.0 * sum(dic_string_disparate[each]) / len(dic_string_disparate[each])) + ','
                        + str(statistics.stdev(dic_string_meandiff[each])) + ','
                        + str(statistics.stdev(dic_string_accuracy[each])) + ','
                        + str(statistics.stdev(dic_string_recall[each])) + ','
                        + str(statistics.stdev(dic_string_precision[each])) + ','
                        + str(statistics.stdev(dic_string_f1[each])) + ','
                        + str(statistics.stdev(dic_string_false[each])) + ','
                        + str(statistics.stdev(dic_string_equopp[each])) + ','
                        + str(statistics.stdev(dic_string_statis[each])) + ','
                        + str(statistics.stdev(dic_string_averageodd[each])) + ','
                        + str(statistics.stdev(dic_string_disparate[each])) + ','
                                                                              '\n')
    writefile.close()

def drawFig(datasetname,protectedattribute,filepath):
    newpath = filepath.replace('.csv','_average.csv')

    readfile = open(newpath)
    lines = readfile.readlines()
    testaccu_list = []
    equaloppfairmetric_list = []
    statislist = []
    averageoddlist = []
    disparate_impactlist = []
    traindifflist = []

    divia_add_testaccu_list = []
    divia_add_equaloppfairmetric_list = []
    divia_add_statislist = []
    divia_add_averageoddlist = []
    divia_add_disparate_impactlist = []
    divia_add_traindifflist = []

    divia_sub_testaccu_list = []
    divia_sub_equaloppfairmetric_list = []
    divia_sub_statislist = []
    divia_sub_averageoddlist = []
    divia_sub_disparate_impactlist = []




    for thisline in lines:
        print(thisline)
        if 'trainsizeratio' in thisline:
            continue
        splits = thisline.split(',')
        feature = splits[1]
        if feature == '1' or feature == '0':
            continue
        traindifflist.append((float(splits[3])))
        testaccu_list.append((float(splits[4])))
        equaloppfairmetric_list.append((float(splits[5])))
        statislist.append((float(splits[6])))
        averageoddlist.append((float(splits[7])))
        disparate_impactlist.append((float(splits[8])))


        divia_add_testaccu_list.append((float(splits[12])+float(splits[4])))
        divia_add_equaloppfairmetric_list.append((float(splits[13])+float(splits[5])))
        divia_add_statislist.append((float(splits[14])+float(splits[6])))
        divia_add_averageoddlist.append((float(splits[15])+float(splits[7])))
        divia_add_disparate_impactlist.append((float(splits[16])+float(splits[8])))


        divia_sub_testaccu_list.append(-(float(splits[12]) - float(splits[4])))
        divia_sub_equaloppfairmetric_list.append(-(float(splits[13]) - float(splits[5])))
        divia_sub_statislist.append(-(float(splits[14]) - float(splits[6])))
        divia_sub_averageoddlist.append(-(float(splits[15]) - float(splits[7])))
        divia_sub_disparate_impactlist.append(-(float(splits[16]) - float(splits[8])))


    range_ = np.arange(0.1,1.0,0.1)
    fig, ax1 = plt.subplots(figsize=(4, 6))

    lines += ax1.plot(range_, statislist, '.-.', color='b', label='statistical parity', linewidth=5)
    ax1.fill_between(range_, divia_add_statislist, divia_sub_statislist, facecolor='b', alpha=0.1)

    lines += ax1.plot(range_, averageoddlist, '--', color='black', label='average abs odds', linewidth=5)
    ax1.fill_between(range_, divia_add_averageoddlist, divia_sub_averageoddlist, facecolor='orange', alpha=0.1)

    lines += ax1.plot(range_, equaloppfairmetric_list, '-', marker='o', color='r', label='equal opportunity',
                      linewidth=5, markersize=9)
    ax1.fill_between(range_, divia_add_equaloppfairmetric_list, divia_sub_equaloppfairmetric_list, facecolor='r',
                     alpha=0.1)

    lines += ax1.plot(range_, disparate_impactlist, ':', color='green', label='disparate impact', linewidth=7)
    ax1.fill_between(range_, divia_add_disparate_impactlist, divia_sub_disparate_impactlist, facecolor='green',
                     alpha=0.1)
    ax1.set_title(datasetname + ' - ' + protectedattribute.lower(), fontsize=25, fontweight='bold')
    ax1.set_xlabel('', fontsize=28, fontweight='bold')
    ax1.set_ylabel('', color='black', fontsize=28, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=25)
    ax1.yaxis.set_tick_params(labelsize=28)

    ax1.set_ylim((-0.1, 1.0))
    from matplotlib.ticker import FormatStrFormatter
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.xticks(np.arange(min(range_), max(range_) + 0.01, 0.2))
    plt.savefig('./../plots/'+datasetname+'-'+protectedattribute+'-ds.pdf',bbox_inches='tight')

    plt.show()

def getdata_credit():
    datasetfilepath = './dataset/CreditCardClients.csv'

    dataset = pd.read_csv(datasetfilepath)

    proc_dataset= dataset.replace(['male','female','high school','higher ed'],[1,0,1,0])

    protected_attribute = ['sex']

   # proc_dataset = fit_string_data(proc_dataset,['Position'])

    dataset_orig_bin = aif360.datasets.BinaryLabelDataset(favorable_label=1,
                                                                unfavorable_label=0,
                                                                df=proc_dataset,
                                                                label_names=['default'],
                                                                protected_attribute_names=protected_attribute)

    return dataset_orig_bin

if __name__ == '__main__':
    df,privileged_groups,unprivileged_groups = get_data('adult','sex')
    print(df)


