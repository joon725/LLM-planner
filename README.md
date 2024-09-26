## LLM planner 
- By Seokjoon Kwon from control lab, KAIST

### 1. Install ollama

Install ollama for Linux

Link : https://ollama.com/download

### 2. Install llama

After installing ollama, install llama with a command below

``` shell script
ollama pull llama3.1:8b
```

Currently, llama3.1 has been released and llama3.1:8b(4.7GB) is used in this framework.
You have options to choose llama models by your preference with commands below.

``` shell script
# llama3.1:70b - llama3.1 with 70b number of parameters. Model size is 40GB.
ollama pull llama3.1:70b
# llama3.1:405b - llama3.1 with 405b number of parameters. Model size is 229GB.
ollama pull llama3.1:405b
# llama3:8b - llama3 with 8b number of parameters. Model size is 4.7GB.
ollama pull llama3
# llama3:70b - llama3 with 70b number of parameters. Model size is 40GB. 
ollama pull llama3:70b
```

For more information about usable models please refer to the below link. Models can be searched in 'Search models' search box 

Link : https://ollama.com/

### 3. Install llava

Install llava for scene graph generation.

``` shell script
ollama pull llava
```

Currently, llava:7b(4.7GB) is used in this framework.
You have options to choose llama models by your preference with commands below.

``` shell script
# llava:13b - llava with 13b number of parameters. Model size is 8GB.
ollama pull llava:13b
# llava:34b - llava with 34b number of parameters. Model size is 20GB.
ollama pull llama3.1:405b
```

### 4. Download BLIP pretrained weight

BLIP pretrained weight should be downloaded. BLIP is used for visual/text feature extractor. Download link is shown below.

Link : https://github.com/salesforce/BLIP?tab=readme-ov-file

In the link, please find "Pre-trained checkpoints" from the README screen, and click "Download" from "BLIP w/ ViT-B" column, "129M" row.
The name of the checkpoint file is model_base.pth

### 5. Locate YOLO weights and BLIP weights.

Please create "weights" folder in the source code files.
Then create two folders named as "BLIP", "YOLO".
In the two folders, pretrained weights for each mdoel should be contained.
For BLIP, "model_base.pth" from the step4 should be contained in "weights/BLIP" folder.
For YOLO, AI2THOR_total.pt, best.pt that were delivered through email, should be contained in "weights/YOLO" folder.
- AI2THOR_total.pt : pretrained weights for custom AI2THOR image dataset.
- best.pt : pretrained weights for normal coco dataset.

Detailed directory of the above explanation is below.


```
root                                  
├── cfg
├── BLIP
...
├── weights
    ├── BLIP
        ├── model_base.pth
    ├── YOLO
        ├── AI2THOR_total.pt
        ├── best.pt 
```

### 4. Install python & pytorch

Install python3.8 & pytorch version 2.0.1(GPU version).

Currently, the framework is being developed under pytorch 2.0.1+cu117(cuda 11.7 version)

Please check your cuda version first, and then install a proper pytorch for the cuda version.

pytorch 2.0.1 can be installed in the below link.

Link : https://pytorch.org/get-started/previous-versions/

* caution : if you use conda version, you should use commands that start with 'pip' to properly install a pytorch with cuda being available.

### 5. Install the packages included in requirements.txt

``` shell script
pip install -r requirements.txt 
```

### 6. Test task_planner.py and environment_recognition_module.py

``` shell script
python task_planner.py
python environment_recognition_module.py
```



