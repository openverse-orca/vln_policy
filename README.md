# 创建虚拟环境

```
conda create -n vln_policy python=3.9 cmake=3.14.0 -y
conda activate vln_policy
```

# 切换目录

假设项目clone在`～/vln_policy`

切换目录至`~/vln_policy`

pip install --proxy http://127.0.0.1:7897加速

# 安装packages

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67 salesforce-lavis==1.0.2
```

然后运行（有显示器的电脑）：

```
conda install habitat-sim=0.2.4 -c conda-forge -c aihabitat -y
```

如果没有显示器:

```
conda install habitat-sim=0.2.4 headless -c conda-forge -c aihabitat -y
```

然后安装vlfm模块：

```
pip install -e .
```

确保timm==0.6.12

安装 habitat-lab :

```
pip install habitat-baselines==0.2.420230405 habitat-lab==0.2.420230405
```

克隆yolov7：

```
git clone https://github.com/WongKinYiu/yolov7.git
```

模型下载，放入OrcaGym/envs/vln_policy/data目录下：

```
mobile_sam.pt:  https://github.com/ChaoningZhang/MobileSAM
groundingdino_swint_ogc.pth：https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
yolov7-e6e.pt: https://github.com/WongKinYiu/yolov7


# 关于数据集下载,ID和SECRET获取地址(通过API获取)： https://my.matterport.com/settings/account/devtools
MATTERPORT_TOKEN_ID=<YOUR TOKEN ID>
MATTERPORT_TOKEN_SECRET=<YOUR TOKEN SECRET>
DATA_DIR=<OrcaGym/envs/vln_policy/data>
HM3D_OBJECTNAV=https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip

wget $HM3D_OBJECTNAV && unzip objectnav_hm3d_v1.zip && mkdir -p $DATA_DIR/datasets/objectnav/hm3d && mv objectnav_hm3d_v1 $DATA_DIR/datasets/objectnav/hm3d/v1 && rm objectnav_hm3d_v1.zip

```

# 报错TIPS：

这里如果报错

```

File "/home/<user name>/anaconda3/envs/vln_policy/lib/python3.9/site-packages/scipy/interpolate/_fitpack_impl.py", line 103, in <module>
'iwrk': array([], dfitpack_int), 'u': array([], float),
TypeError

```

可以多卸载几次numpy， 然后再安装numpy, 参考https://github.com/Genesis-Embodied-AI/Genesis/issues/117

```

pip uninstall numpy -y #多卸载几次
pip install numpy==1.26.4

```

# （选择）关于habitat数据集下载

ID和SECRET获取地址(通过API获取)：

`https://my.matterport.com/settings/account/devtools`

```

MATTERPORT_TOKEN_ID=7dbd513432b07ece
MATTERPORT_TOKEN_SECRET=c12e721e6e72a62ae8aa44330d9a4e65
DATA_DIR=/home/<your pc name>/OrcaGym/envs/vln_policy/data
HM3D_OBJECTNAV=https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip

```

## 然后下载数据集

(这里场景,就是hm3d_train_v0.2和hm3d_val_v0.2, 下载一定要根据下面的python格式进行下载)：

这里的train训练集可以不下

```

python -m habitat_sim.utils.datasets_download --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET --uids hm3d_train_v0.2  --data-path $DATA_DIR
python -m habitat_sim.utils.datasets_download --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET --uids hm3d_val_v0.2   --data-path $DATA_DIR

```
