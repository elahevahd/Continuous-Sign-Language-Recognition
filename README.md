STEP 0. Unzip "container.tar.gz". 

STEP 1. Setting Docker environment. 

1.1. The docker installation:  https://docs.docker.com/install/linux/docker-ce/ubuntu/

1.2. The nvidia-docker installation: https://chunml.github.io/ChunML.github.io/project/Installing-NVIDIA-Docker-On-Ubuntu-16.04/
or https://github.com/NVIDIA/nvidia-docker

1.3. Loading docker Images, run this commmand: sudo docker load --input ./NSF_LEARN_Docker.tar

1.4 Create a container using the following commands (3 GPUs are needed):

sudo docker run -it --rm --runtime=nvidia -v /home/:/home/ --log-opt max-size=50m  -e NVIDIA_VISIBLE_DEVICE=0,1,2 cwaffles/openpose

cd ./container/openpose 

or enter the path to the folder "container/openpose"

STEP 2. Running the code.

2.1. Find the following line in code_asl.py:  

dropbox_vid_folder = '/home/ultra-server/Dropbox/NSF-LEARN/video/'    

This is the path to the folder with upcoming videos. Any video that will be uploaded to this folder will be processed by the code.

The path can be changed to any dropbox folder that is synced with the machine. 

2.2. Enter this command to run the code: python code_asl.py

2.3. The CSV file will be generated in the same directory as the one in 1.5 ('/home/ultra-server/Dropbox/NSF-LEARN/video/'). 

STEP 3. The feedback software only works on Windows. 

3.1. Extract Feedback Reviewer zip file.  

3.2 Open the file pathname_video.txt and paste the path to *your Dropbox path* (example: 'C:\Users\ASLLAB\Dropbox\NSF-LEARN\video')

This path should point to the same dropbox directory as the one in 2.1 ('Dropbox/NSF-LEARN/video/')

3.3. Enter the Log In details in the file classlist_information.txt. Each line includes the Name and ID of one student. Example:

Test1,123456
Test2,0001111

3.4. Run the demo1 file: 

A window pops up: enter the Name and ID of the student identical to the one in classlist_information.txt. 

Name: Test1
ID: 123456

3.5. Click on Sign in: you see a list of videos. You can select any of the videos to see the video and feedback. Press "Review". 



