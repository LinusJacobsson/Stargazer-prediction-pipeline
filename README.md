# DE2 Final Project - A Stargazer prediction pipeline

<p align="center">
  <img src="banner.png" alt="Banner Image">
</p>

This project was performed in the [Data Engineering 2 course](https://www.uu.se/en/admissions/freestanding-courses/course-syllabus/?kKod=1TD075&lasar=) at Uppsala University. Our aim was to build a complete machine learning pipeline, enabling the launch and support of a prediction based application for estimating the number of stargazers of different GitHub repositories. 


### Prerequisites

OpenStack APIs - only needed on the client machine for contextualization. Instructions can be found [here](https://github.com/sztoor/model_serving.git)

GitHub API - optional since the data for this project is already in the repo. 



## Getting Started

After creating an initial VM's to use a a base. Clone this [repo](https://github.com/sztoor/model_serving.git). This contans the skeleton from which we built our project. Navigate to the single_node_with_docker_ansible_client directory on the client server and run

´´´python
python3 start_instances.py
´´´
Note: you have to modify the setup variables in start_instances.py
´´´bash
git clone https://github.com/LinusJacobsson/Stargazer-prediction-pipeline
´´´
Navigate to the server hosting
´´´bash
cd Stargazer-prediction-pipeline/production server
´´´
Start the docker containers
´´´bash
sudo docker-compose build
sudo docker-compose up -d
´´´

Check that everything works well using
´´´bash
sudo docker ps
´´´
and that the welcome page is visible at: http://<PUBLIC-IP-ADRESS>:5100

After configuring the IaC scripts and succesfully launching the development and production servers, clone this repository into both. After that add the update.sh script as a post-commit hoook on the development server. Now everything is pretty much set up as intented. 
Now the models can simply be improved on the development server, and any commit will test for the best model and push it to the production server.

