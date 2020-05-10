# RefineDet
`
CopyRight:KeyanChen
Version: v1.1   
Date: 20200417    
`


RefineDet Detection Network Based On Pytroch. 

It involves SSD.   

It is developed by myself!
# 文件解读

1. Config:配置文件
2. Train_RefineDet:训练脚本
3. Test_Full_Tiff:测试带地理信息的Tiff图
4. Test_Jpg_Txt:测试小图，生成测试的bbox信息到txt中

# 训练修改
1. 修改Config文件内容
2. 使用tools工具集获取必要的参数
2. utils/Augmentation中所需的增广内容

# Version: v1.1
1. 加入Tiny_Scale_Net，具体的在basenet上增加了一个128的检测头，存在于nets/RefineDet_TS中
2. 加入Focal_loss，存在于nets/layers/Focal_Loss中
3. 检测时加入score阈值和非极大值抑制的iou阈值