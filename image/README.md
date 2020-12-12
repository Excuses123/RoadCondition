# 镜像
#### 镜像说明

从百度云盘下载镜像打包文件：baokemeng.tar
百度云盘地址: https://pan.baidu.com/s/14AX72R9zBfb04fPs6vHdWQ  密码: vqa3

把下载得到的baokemeng.tar放到您linux系统的某个目录下，然后在这个目录下，执行以下命令，把镜像打包文件导入到您本机中成为一个镜像。
`sudo docker load -i baokemeng.tar`

然后基于该镜像生成容器并开始跑代码的命令如下：
`sudo nvidia-docker run -v /data:/data baokemeng:1 sh run.sh`
