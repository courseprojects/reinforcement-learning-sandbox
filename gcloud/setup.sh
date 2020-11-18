sudo apt update
sudo apt -y install apt-transport-https ca-certificates curl gnupg2 software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/debian \
   $(lsb_release -cs) \
   stable"
deb [arch=amd64] https://download.docker.com/linux/debian buster stable
sudo apt update
sudo apt -y install docker-ce docker-ce-cli containerd.io
sudo apt -y install git


sudo git clone https://github.com/peterdavidfagan/CS221-Project.git
sudo chmod -R a+rwx ./CS221-Project
cd ./CS221-Project/docker/
sudo gsutil cp -r gs://robot-license/mjkey.txt ./
sudo docker build -t robosuite .
sudo docker run -it -d --gpus all --name robosuite robosuite:latest
