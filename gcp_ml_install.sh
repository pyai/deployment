#!/bin/bash
## COPYRIGHT
## 
## All contributions by Yu Chun Yang (https://github.com/exactone):
## Copyright (c) 2015 - 2030, Yu Chun Yang.
## All rights reserved.
##
## Each contributor holds copyright over their respective contributions.
## The project versioning (Git) records all such contribution source information.

# 手動設定安裝步驟數量
ALLSTEP=10

TF_GPU_CPU='tensorflow-cpu'
tensorflow_install_option(){
	echo "******************************"
	echo "which version of tensorflow would you like to install ? (請問您想要安裝哪個版本的tensorflow?)"
	echo "Press c for CPU, g for GPU. (安裝CPU版本請按 c, GPU版本請按g)"
	echo "******************************"
	read TF_GPU_CPU
	if [ "$TF_GPU_CPU" == "c" -o "$TF_GPU_CPU" == "C" ]; then
		TF_GPU_CPU='tensorflow-cpu'
	elif [ "$TF_GPU_CPU" == "g" -o "$TF_GPU_CPU" == "G" ]; then
		TF_GPU_CPU='tensorflow-gpu'
	else
		echo "Illegal setting, tensorflow-CPU version assigned choosed. (錯誤的設定，將安裝tensorflow-CPU)"
		TF_GPU_CPU='tensorflow-cpu'
	fi

	# 安裝 tensorflow-cpu 比安裝tensorflow-gpu 少了 "安裝cuda 9.0", "安裝cudnn 7.0" 2步驟
	if [ "$TF_GPU_CPU" == 'tensorflow-cpu' ]; then
		ALLSTEP=$[$ALLSTEP-2]
	fi
}
tensorflow_install_option
echo $TF_GPU_CPU

STEP=1
echo_progress () {
	echo "=============================="
    echo "step $STEP / $ALLSTEP"
    echo $1 $2
    echo "=============================="
    STEP=$[$STEP+1]
}

PWD=`pwd`
cd ~

##########
echo_progress "seeting sudo password" "(設定sudo密碼)"
##########
echo "whoami:`whoami`"
sudo passwd
cd ~

##########
# 依自己的設定設置swap大小，單位MB, 建議8192
SWAPSIZE = 20
echo_progress "setting swap, swap size: $SWAPSIZE MB" "(設定 swap, swap大小為: $SWAPSIZE MB)"
##########
# 使用 dd 指令建立 Swap File
sudo dd if=/dev/zero of=/swapfile bs=1M count=$SWAPSIZE

# 格式化 Swap File
sudo mkswap /swapfile
sudo chmod 0600 /swapfile
# 開機掛載swap設定
echo "/swapfile    swap    swap    defaults    0 0" >> /etc/fstab
# 啟動 swap
swapon /swapfile
cd ~

##########
echo_progress "SSL authentication, manual enter needed" "(設定 SSL授權金鑰, 請手動輸入金鑰認證資訊)"
##########
mkdir ~/ssl
cd ~/ssl
openssl req -x509 -nodes -days 3650 -newkey rsa:1024 -keyout ~/ssl/mykey.key -out ~/ssl/mycert.pem
cd ~

##########
echo_progress "install htop, tmux, p7zip, bzip2" "(安裝 htop, tmux, p7zip, bzip2)"
##########
sudo apt-get update
sudo apt-get install -y htop
sudo apt-get install -y tmux
sudo apt-get install -y p7zip-full
sudo apt-get install -y bzip2

if [ "$TF_GPU_CPU" == "tensorflow-gpu" ]; then
	##########
	echo_progress "install cuda 9.0 for TensorFlow 1.6" "(安裝cuda 9.0, TensorFlow 1.6支援cuda 9.0)"
	##########
	mkdir ~/download
	cd ~/download
	wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb

	sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
	sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
	sudo apt-get update -y
	yes Y | sudo apt-get install cuda-libraries-9-0

	export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
	export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

	echo "export PATH=$PATH" >> ~/.bashrc
	echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/.bashrc

	# tensorflow 專用變數 CUDA_HOME
	export CUDA_HOME=/usr/local/cuda-9.0
	echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc

	##########
	echo_progress "install cudnn 7 for TensorFlow 1.6" "(安裝cudnn 7, cudnn 7支援cuda 9.0)"
	##########
	cd ~/download
	wget -O libcudnn7_7.1.1.5-1+cuda9.0_amd64.deb "https://www.dropbox.com/s/1ac058twuvn9x21/libcudnn7_7.1.1.5-1%2Bcuda9.0_amd64.deb?dl=0"
	yes Y | sudo dpkg -i libcudnn7_7.1.1.5-1+cuda9.0_amd64.deb
	#sudo apt-get install libcupti-dev
	cd ~

elif [ "$TF_GPU_CPU" == "tensorflow-cpu" ]; then
	mkdir ~/download
	cd ~
fi
##########
## echo_progress "install cuda 9 for TensorFlow 1.6" "(安裝cuda 9.1, TensorFlow 1.6支援cuda 9)"
##########
## mkdir ~/download
## cd ~/download
## wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
## sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
## sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
## sudo apt-get update -y
## yes Y | sudo apt-get install cuda

## export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
## export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
## echo "export PATH=$PATH" >> ~/.bashrc
## echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ~/.bashrc


##########
## echo_progress "install cudnn 7 for TensorFlow 1.6" "(安裝cudnn 7, cudnn 7支援cuda 9.1)"
##########
## cd ~/download
## wget https://www.dropbox.com/s/6ps72ftw1e7rpd9/libcudnn7_7.1.1.5-1%2Bcuda9.1_amd64.deb
## yes Y | sudo dpkg -i libcudnn7_7.1.1.5-1+cuda9.1_amd64.deb
## sudo apt-get install libcupti-dev
## cd ~

##########
echo_progress "install anaconda" "(安裝anaconda)"
##########
cd ~/download
rm -f Anaconda*x86_64.sh
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
ANACONDA=`realpath ./Anaconda*x86_64.sh`
chmod 755 $ANACONDA
$ANACONDA -b -p ~/anaconda3
export PATH=$HOME/anaconda3/bin:$PATH
echo "export PATH=$PATH" >> ~/.bashrc

# conda update -n base conda
cd ~

##########
echo_progress "set jupyter notebook server, port: 9999" "(設定 jupyter notebook server, port: 9999)"
##########
mkdir ~/jupyter_server
yes y | jupyter notebook --generate-config
sedhome=$(echo $HOME | sed 's/\//\\\//g')
sed -i "s/#c.NotebookApp.certfile = ''/c.NotebookApp.certfile = '$sedhome\/ssl\/mycert.pem'/g"  ~/.jupyter/jupyter_notebook_config.py 
sed -i "s/#c.NotebookApp.keyfile = ''/c.NotebookApp.keyfile = '$sedhome\/ssl\/mykey.key'/g"  ~/.jupyter/jupyter_notebook_config.py
sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '\*'/g"  ~/.jupyter/jupyter_notebook_config.py
touch get_sha_passwd.py
echo "from IPython.lib import passwd" >> get_sha_passwd.py
echo "print(passwd())" >> get_sha_passwd.py
sha1passwd=`python get_sha_passwd.py`
rm -f get_sha_passwd.py
sed -i "s/#c.NotebookApp.password = ''/c.NotebookApp.password = u'$sha1passwd'/g" ~/.jupyter/jupyter_notebook_config.py 
sed -i "s/#c.NotebookApp.open_browser = True/c.NotebookApp.open_browser = False/g" ~/.jupyter/jupyter_notebook_config.py
sed -i "s/#c.NotebookApp.port = 8888/c.NotebookApp.port = 9999/g" ~/.jupyter/jupyter_notebook_config.py

##########
echo_progress "install keras, $TF_GPU_CPU" "(安裝 keras, $TF_GPU_CPU)"
##########
#source ~/.bashrc
yes y | conda update conda
yes Y | conda create --name keras anaconda
source activate keras
pip install keras

if [ "$TF_GPU_CPU" == "tensorflow-cpu" ]; then
	pip install tensorflow
else
	pip install tensorflow-gpu
fi

##########
echo_progress "install kaggle-cli" "(安裝 kaggle-cli)"
##########
pip install --upgrade pip
pip install kaggle-cli
source deactivate
