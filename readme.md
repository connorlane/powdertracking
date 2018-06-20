### Powder Particle Tracking for Blown-Powder Laser Additive Manufacturing ###

Software for tracking powder particles as they collide with a planar horizontal
substrate in high-speed video. Uses a RANSAC-based algorithm to identify 
particle trajectories in the video and detect the particles bouncing off a
substrate.

#### Quick Start ####
1. Clone repo:
```git clone git@github.com:connorlane/powdertracking.git```
2. CD into repo:
```cd powdertracking```
3. Establish virtual environment::
```virtualenv env```
4. Source virtual environment:
```source env/bin/activate```
5. Install python dependencies:
```pip install -r requirements.txt```
6. Run main.py demo:
```python main.py data/samplevideo.mp4```
7. Visualize (data visualization example):
```python visualize.py data/samplevideo.mp4```
8. Generate heatmap (data visualization example):
```python heatmap.py samplevideo.csv```

