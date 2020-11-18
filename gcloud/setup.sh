sudo git clone https://github.com/peterdavidfagan/CS221-Project.git
sudo chmod -R a+rwx ./CS221-Project
cd ./CS221-Project/docker/
sudo gsutil cp -r gs://robot-license/mjkey.txt ./
sudo docker build -t robosuite .
sudo docker run -it -d --gpus all --name robosuite robosuite:latest
