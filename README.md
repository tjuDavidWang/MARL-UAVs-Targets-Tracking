# MARL-UAVs-Targets-Tracking
The implement and improvement of the paper “Improving multi-target cooperative tracking guidance for UAV swarms using multi-agent reinforcement learning”.

![](https://github.com/tjuDavidWang/MARL-UAVs-Targets-Tracking/blob/main/imgs/2d-demo.png)

### Environment

You can simply use pip install to config the environment:

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm tensorboard scipy
pip install imageio[ffmpeg]
```

### Run the code

```sh
cd src
python main.py
```

### ToDo List

- [x] vanilla MAAC
  - [x] Actor-Critic framework
- [x] MAAC-R
  - [x] reciprocal reward (with PMI network)
- [x] MAAC-G
  - [x] receive the global reward
- [x] 3D demo
