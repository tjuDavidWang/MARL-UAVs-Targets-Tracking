# MARL-UAVs-Targets-Tracking
The implement and improvement of the paper “Improving multi-target cooperative tracking guidance for UAV swarms using multi-agent reinforcement learning”.

### Environment

You can simply use pip install to config the environment:

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm
```

### Run the code

```sh
python actor-critic.py
```

### ToDo List

- [x] Actor-Critic framework
- [x] 2D demo
- [ ] reciprocal reward (with PMI network)
- [ ] 3D demo