# Continuous-Sign-Language-Recognition
### Implementation of '[Recognizing American Sign Language Nonmanual Signal Grammar Errors in Continuous Videos](https://arxiv.org/pdf/2005.00253.pdf)' (ICPR 2020)

> **Recognizing American Sign Language Nonmanual Signal Grammar Errors in Continuous Videos**<br>
> Elahe Vahdani, Longlong Jing, Yingli Tian, Matt Huenerfauth
>
> Paper: https://arxiv.org/pdf/2005.00253.pdf
>
> **Abstract:** *As part of the development of an educational tool that can help students achieve fluency in American Sign Language (ASL) through independent and interactive practice with immediate feedback, this paper introduces a near real-time system to recognize grammatical errors in continuous signing videos without necessarily identifying the entire sequence of signs. Our system automatically recognizes if a performance of ASL sentences contains grammatical errors made by ASL students. We first recognize the ASL grammatical elements including both manual gestures and nonmanual signals independently from multiple modalities (i.e. hand gestures, facial expressions, and head movements) by 3D-ResNet networks. Then the temporal boundaries of grammatical elements from different modalities are examined to detect ASL grammatical mistakes by using a sliding window-based approach. We have collected a dataset of continuous sign language, ASL-HW-RGBD, covering different aspects of ASL grammars for training and testing. Our system is able to recognize grammatical elements on ASL-HW-RGBD from manual gestures, facial expressions, and head movements and successfully detect 8 ASL grammatical mistakes.*


### Setting Docker Environment
1. The docker installation:  https://docs.docker.com/install/linux/docker-ce/ubuntu/
2. The nvidia-docker installation: https://chunml.github.io/ChunML.github.io/project/Installing-NVIDIA-Docker-On-Ubuntu-16.04/ or https://github.com/NVIDIA/nvidia-docker
3. Loading docker Images, run this commmand: sudo docker load --input ./NSF_LEARN_Docker.tar
4. Create a container using the following commands (3 GPUs are needed):
    sudo docker run -it --rm --runtime=nvidia -v /home/:/home/ --log-opt max-size=50m  -e NVIDIA_VISIBLE_DEVICE=0,1,2 cwaffles/openpose
    cd ./container/openpose 
    or enter the path to the folder "container/openpose"

### Running
1. Find the following line in code_asl.py: dropbox_vid_folder = '/home/ultra-server/Dropbox/NSF-LEARN/video/'    
   This is the path to the folder with upcoming videos. Any video that will be uploaded to this folder will be processed by the code.
   The path can be changed to any dropbox folder that is synced with the machine. 
2. Enter this command to run the code: python code_asl.py
3. The CSV file will be generated in the same directory as the one in 1.5 ('/home/ultra-server/Dropbox/NSF-LEARN/video/'). 

### The Feedback Software
  (This software only works on Windows.)
1. Extract Feedback Reviewer zip file.  
2 Open the file pathname_video.txt and paste the path to *your Dropbox path* (example: 'C:\Users\ASLLAB\Dropbox\NSF-LEARN\video')
  This path should point to the same dropbox directory as the one in 2.1 ('Dropbox/NSF-LEARN/video/')
3. Enter the Log In details in the file classlist_information.txt. Each line includes the Name and ID of one student. Example:
  Test1,123456
  Test2,0001111
4. Run the demo1 file: 
  A window pops up: enter the Name and ID of the student identical to the one in classlist_information.txt. 
  Name: Test1
  ID: 123456
5. Click on Sign in: you see a list of videos. You can select any of the videos to see the video and feedback. Press "Review". 



