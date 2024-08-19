# Co-Simulation of Power Grid and Communication Network
## for the Master Thesis of Moritz Volkmann

### To run the GridSim, do the following
```bash
pip install -r requirements.txt
python3 GridSim.py
```

### To run the NetSim, do the following:

1. If you already installed ns3, you should add a symlink to its directory to the ext directory like this:
```bash
ln -s PATH/TO/ns3 ext/ns3
```
If not it should (hopefully) be installed while cloning the repository and put there

2. Configure CMake Project
3. Build and run scratch_NetSim or scratch_test


### NetSim.py is deprecated and does not work properly, since some of the functionalities did not work in the ns-3 Python version

This is how the Program should work once it is finished:

![Sequence Diagram](./figures/CoSim.png)

Or like this: (not yet decided)

![Sequence Diagram 2](./figures/CoSimv2.png)
