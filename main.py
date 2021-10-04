from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import scipy.io as scio
import numpy as np
from minisom import MiniSom
import math
import torch
from RBF import RBF
from sklearn import svm

'''
分类函数
'''
def classify(som,data,winmap):
    from numpy import sum as npsum
    default_class = npsum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

def SOM_RBF():
    '''
    导入数据 214个1 116个-1
    '''
    data_train = scio.loadmat('data_train.mat')['data_train']
    data_test = scio.loadmat('data_test.mat')['data_test']
    data_label = scio.loadmat('label_train.mat')['label_train'].squeeze()
    print(data_train)
    '''
    #划分训练集和验证集 7:3
    '''
    X_train, X_valid, y_train, y_valid = train_test_split(data_train,data_label,test_size=0.3,random_state=1,stratify=data_label)
    print(sum(y_train))
    print(">> shape of training data:",X_train.shape)
    print(">> shape of validation data:",X_valid.shape)

    '''
    #训练SOM
    '''
    N = X_train.shape[0]  #样本数量
    M = X_train.shape[1]  #维度/特征数量

    '''
    设置超参数
    '''
    # size = math.ceil(np.sqrt(5 * np.sqrt(N))) # 经验公式：决定输出层尺寸
    size=8
    print("训练样本个数:{}  测试样本个数:{}".format(N,X_valid.shape[0]))
    print("输出网格最佳边长为:",size)
    max_iter = 40000
    sigma_0= (np.sqrt((size-1)**2+(size-1)**2))/2
    print('initial sigma:',sigma_0)
    # Initialization and training
    som = MiniSom(size, size, M, sigma=1, learning_rate=0.1, neighborhood_function='gaussian')

    '''
    初始化权值，有2个API
    '''
    #som.random_weights_init(X_train)
    som.pca_weights_init(X_train)

    '''
    开始训练
    '''
    som.train_batch(X_train, max_iter, verbose=False)
    #som.train_random(X_train, max_iter, verbose=False)

    '''
    分类
    '''
    winmap = som.labels_map(X_train,y_train)
    print(winmap)

    y_pred = classify(som,X_valid,winmap)
    print('SOM validation accuracy')
    print(classification_report(y_valid, np.array(y_pred)))

    '''
    输出权重
    '''
    weights = som.get_weights().copy()
    # print(type(weights))
    # print(weights.shape)
    # print(weights)
    center_vectors=weights.reshape(-1,33)
    print(">> shape of center_vectors:",center_vectors.shape)
    # print(center_vectors)

    '''
    # 训练RBF
    # '''
    #转换数据类型X_train, X_valid, y_train, y_valid, data_test
    center_vectors = torch.tensor(center_vectors)
    X = torch.tensor(X_train)
    # print(X)
    train_label = torch.tensor(y_train).unsqueeze(1)
    y_trainT = torch.tensor(y_train,dtype=torch.int32)
    # print(y_trainT.dtype)
    valid_X = torch.tensor(X_valid)
    valid_label = torch.tensor(y_valid).unsqueeze(1)
    y_validT = torch.tensor(y_valid, dtype=torch.int32)
    test_X = torch.tensor(data_test)
    # print(test_X)
    # print(center_vectors.dtype,X.dtype)
    # print(X,X.size())
    # print(train_label,train_label.size())

    print('Building RBF model...')
    RBF_model = RBF(33, center_vectors, 1)
    weight = RBF_model.train(X, train_label)

    predictions_train = RBF_model.test(X).int()
    predictions_valid = RBF_model.test(valid_X).int()
    predictions_test = RBF_model.test(test_X).int()
    # print(predictions_train.dtype)
    # print(predictions_valid)
    # print(y_validT)
    print(classification_report(y_trainT,predictions_train))
    print(classification_report(y_validT,predictions_valid))
    accuracy_train = (predictions_train == y_trainT).sum() / torch.tensor(X.size(0)).float()
    accuracy_valid = (predictions_valid == y_validT).sum() / torch.tensor(valid_X.size(0)).float()
    print("accuracy of training data:", accuracy_train)
    print("accuracy of validation data:", accuracy_valid)
    print(predictions_test)

    # '''
    # 可视化
    # '''
    # # U-max
    # # plt.figure(figsize=(9,9),num=0)
    # # heatmap = som.distance_map()  #生成U-Matrix
    # # print(heatmap,heatmap[3,1])
    #
    # # plt.imshow(heatmap,cmap='bone_r')
    # # plt.pcolor(heatmap, cmap='bone_r')      #miniSom案例中用的pcolor函数,需要调整坐标
    # # plt.colorbar()
    # # plt.plot(0 + .5, 0 + .5, 'o', markerfacecolor='None',
    # #          markeredgecolor='C0', markersize=12, markeredgewidth=2)
    # # plt.show()
    #
    # plt.figure(figsize=(9, 9),num=0)
    # # 背景上画U-Matrix
    # heatmap = som.distance_map()
    # plt.pcolor(heatmap.T, cmap='bone_r')  # plotting the distance map as background
    # plt.colorbar()
    # # 定义不同标签的图案标记
    # markers = {1: 'o', -1: 's'}
    # colors = {1: 'C0', -1: 'C3'}
    # category_color = {'Class 1': 'C0',
    #                   'Class -1': 'C3',}
    # for cnt, xx in enumerate(X_train):
    #     w = som.winner(xx)  # getting the winner
    #     # 在样本Heat的地方画上标记
    #     plt.plot(w[0]+.5, w[1]+.5, markers[y_train[cnt]], markerfacecolor='None',
    #              markeredgecolor=colors[y_train[cnt]], markersize=12, markeredgewidth=2)
    # plt.axis([0, 8, 0, 8])
    # # ax = plt.gca()
    # # ax.invert_yaxis() #颠倒y轴方向
    # legend_elements = [Patch(facecolor=clr,
    #                          edgecolor='w',
    #                          label=l) for l, clr in category_color.items()]
    # plt.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1.20, 0.9))
    # plt.show()
    #
    # label_name_map_number = {"Class 1": 0, "Class -1": 1}
    #
    # from matplotlib.gridspec import GridSpec
    # plt.figure(figsize=(9, 9),num=1)
    # the_grid = GridSpec(8, 8)
    # # YY=0
    # for position in winmap.keys():
    #     # print(position)
    #     label_fracs = [winmap[position][label] for label in [-1, 1]]
    #     # print(label_fracs)
    #     # YY+=label_fracs[0]
    #     plt.subplot(the_grid[7-position[1], position[0]], aspect=1)
    #     patches, texts = plt.pie(label_fracs,colors=['C3','C0'])
    #     plt.text(position[0] / 100, position[1] / 100, str(len(list(winmap[position].elements()))),
    #              color='black', fontdict={'weight': 'bold', 'size': 15},
    #              va='center', ha='center')
    # # plt.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1.15, 0.95))
    # plt.show()
    # # print(YY)

def SVM():
    data_train = scio.loadmat('data_train.mat')
    data_test = scio.loadmat('data_test.mat')
    data_label = scio.loadmat('label_train.mat')
    X = data_train['data_train']
    y = np.squeeze(data_label['label_train'])
    test = data_test['data_test']
    print(test)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=1,stratify=y)
    print(">> shape of training data:", X_train.shape)
    print(">> shape of validation data:", X_valid.shape)
    print(sum(y_valid))
    print('Building SVM model')
    clf = svm.SVC(C=0.5,kernel='rbf',gamma=0.5,decision_function_shape='ovo')
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_train)
    accuracy_training = (predictions==y_train).sum().__float__() / X_train.shape[0]
    predictions_valid = clf.predict(X_valid)
    accuracy_valid = (predictions_valid == y_valid).sum().__float__() / X_valid.shape[0]
    predictions_y = clf.predict(test)
    print(">>accuracy_training :",accuracy_training)
    print(">>accuracy_validation :", accuracy_valid)
    print(predictions_y)


if __name__ == '__main__':
    mode = 0
    if mode == 0:
        SOM_RBF()
    else:
        SVM()