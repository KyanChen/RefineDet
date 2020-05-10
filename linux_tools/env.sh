#!/bin/sh

# python3 Unzip_Files_Server.py
# pip install torch torchvision -i https://pypi.doubanio.com/simple
# 解压


cat WKK* > WKK_Full.zip
zip -F WKK.zip --output WKK_Full.zip -q
unzip -d /home/aistudio/mydata -q WKK_Full.zip
rm -rf *.jpg
# 查看进程
ps -ef
# 查询分区大小
df -hl
# 任务管理器
top
# 输出到log文件
nohup python3 Train_RefineDet.py > results/myout.log 2>&1 &
jobs
# 将后台中的命令调至前台继续运行
fg %job_number
# 将一个在后台暂停的命令，变成在后台继续执行
bg %job_number
# 进程查看命令
ps -aux | grep "test.sh"
# 查看当前目录下的文件数量（不包含子目录中的文件）
ls -l|grep "^-"| wc -l
# 查看当前目录下的文件数量（包含子目录中的文件） 注意：R，代表子目录
ls -lR|grep "^-"| wc -l

mkdir /home/aistudio/.pip
rm /home/aistudio/.pip/pip.conf
echo '[global] \ntimeout = 60\nindex-url = http://pypi.douban.com/simple\ntrusted-host = pypi.douban.com' >> /home/aistudio/.pip/pip.conf
rm /home/aistudio/.condarc
echo 'channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud' >> /home/aistudio/.condarc

# mkdir /home/aistudio/conda_env
# conda create --prefix=/home/aistudio/conda_env python=3.7
conda init bash
source activate /home/aistudio/conda_env


conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
# pip install tensorboardX==2.0 -t /home/aistudio/external-libraries
pip install tensorboardX==2.0
pip install tensorboard
pip install opencv-python
pip install matplotlib
pip install pandas
pip install tqdm


