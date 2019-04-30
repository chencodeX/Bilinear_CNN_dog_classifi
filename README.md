# 使用BCNN网络结构测试细粒度分类效果


关于BCNN ,理论参考文献http://vis-www.cs.umass.edu/bcnn/docs/bcnn_iccv15.pdf 在项目中，先训练了全连接层的参数并保存下来，然后在微调整个模型

     ./bcnn_DD_woft.py  训练全连接层
         bcnn_finetuning.py  微调整个网络
         bcnn_finetuning_predict.py   预测测试样本（百度流浪狗）
         down_pic.py  处理数据的脚本


    utils/create_h5_dataset.py  创建h5数据集  本文一开始使用这种数据格式
          data_loader.py  一个动态的数据加载工具，可以完成目标检测的提取，使用较少的内存空间。并可进行数据扩增
          utils_.py  工具类


bdgod  本目录为pytorch实现的vgg和resnet，支持预训练，bilinearCNN模型，不支持预训练

    bdgod/data_augmentation.py   数据增广文件
          dog_config.py   配置文件
          load_image.py   旧的数据处理工具
          misc.py         预训练模型参数加载工具
          predict_dog.py   预测功能
          resnet.py        定义resnet网络结构
          train_net.py     训练网络
          vggnet.py        定义vgg网络结构
          BilinearCNN.py   定义BCNN网络结构
          
          
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)