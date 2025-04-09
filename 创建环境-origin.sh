conda create -n vln_policy python=3.9 cmake=3.14.0 -y
# 这里要加上cmake版本。

conda activate vln_policy
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67 salesforce-lavis==1.0.2

# 然后运行（有显示器的电脑）：
conda install habitat-sim=0.2.4 -c conda-forge -c aihabitat -y
# 如果没有显示器:
conda install habitat-sim=0.2.4 headless -c conda-forge -c aihabitat -y

# 然后再运行(不用加上[habitat])：
# 这里可能要到 pyproject.toml 下删除：
# "habitat-sim @ git+https://github.com/facebookresearch/habitat-sim.git@v0.2.4",
# 这一行

# 然后安装vlfm模块
pip install -e .

pip install timm==0.6.12
# 安装Spot API
pip install bosdyn-client bosdyn-api six
pip install git+https://github.com/naokiyokoyama/bd_spot_wrapper.git

# 克隆yolov7
git clone https://github.com/WongKinYiu/yolov7.git
# (选择)克隆Grounding DINO
git clone https://github.com/IDEA-Research/GroundingDINO.git

# 安装 habitat-lab :
pip install habitat-baselines==0.2.420230405 habitat-lab==0.2.420230405 
# pip install habitat-baselines==0.2.520230729 habitat-lab==0.2.520230729

# 关于数据集下载,ID和SECRET获取地址(通过API获取)： https://my.matterport.com/settings/account/devtools
MATTERPORT_TOKEN_ID=7dbd513432b07ece
MATTERPORT_TOKEN_SECRET=c12e721e6e72a62ae8aa44330d9a4e65
DATA_DIR=/home/orca3d/vlfm/data
HM3D_OBJECTNAV=https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip

# 然后下载数据集(这里场景,就是hm3d_train_v0.2和hm3d_val_v0.2, 下载一定要根据下面的python格式进行下载)：
sudo chmod 777 -R data/
# 这里的train训练集可以不下
python -m habitat_sim.utils.datasets_download --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET --uids hm3d_train_v0.2  --data-path $DATA_DIR 
python -m habitat_sim.utils.datasets_download --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET --uids hm3d_val_v0.2   --data-path $DATA_DIR  
# 这里如果报错  File "/home/orca3d/anaconda3/envs/vlfm/lib/python3.9/site-packages/scipy/interpolate/_fitpack_impl.py", line 103, in <module>
#     'iwrk': array([], dfitpack_int), 'u': array([], float),
# TypeError
# 多卸载几次numpy， 然后再安装numpy, https://github.com/Genesis-Embodied-AI/Genesis/issues/117
# pip uninstall numpy -y#多卸载几次
# pip install numpy==1.26.4


wget $HM3D_OBJECTNAV && unzip objectnav_hm3d_v1.zip && mkdir -p $DATA_DIR/datasets/objectnav/hm3d && mv objectnav_hm3d_v1 $DATA_DIR/datasets/objectnav/hm3d/v1 && rm objectnav_hm3d_v1.zip
# 注意路径:
# 正确路径:$DATA_DIR/datasets/objectnav/hm3d/v1/这里是train,val,val_mini
# 错误路径:$DATA_DIR/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1/这里是train,val,val_mini

# 如果手动下载并且放到指定目录：
# 下载 HM3D ObjectNav 数据集（任务数据）
# 这部分数据包含导航任务的具体实例信息（如起点、终点等）。
# 该数据集应存储在 $DATA_DIR/datasets/objectnav/hm3d/v1 中。最终的目录结构应该是：
# vlfm/data/
#      └── datasets/
#          └── objectnav/
#              └── hm3d/
#                  └── v1/
#                      └── ... (任务数据文件,val,train,val_mini)
# 链接：https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip

# 下载权重（3个）：
mobile_sam.pt:  https://github.com/ChaoningZhang/MobileSAM
groundingdino_swint_ogc.pth： wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
yolov7-e6e.pt: https://github.com/WongKinYiu/yolov7


