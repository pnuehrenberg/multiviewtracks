# MultiViewTracks

**MVT** integrates the results from structure-from-motion and video-based tracking to triangulate 3D trajectories.  
**MVT** can be used to acquire highly-detailed animal trajectories for behavioral analyses.

You can find a preprint on [biorxiv](https://www.biorxiv.org/content/10.1101/571232v1), where we used **MVT** in diverse aquatic environments.

### Installation

The easiest way to get **MVT** up and running across most platforms is using a [*miniconda*](https://docs.conda.io/en/latest/miniconda.html) environment with jupyter notebooks for python3. Once conda is setup, a virtual environment can be created and activated by running the following code in a terminal (*Anaconda Promt* on Windows). Further, [*pip*](https://pypi.org/project/pip/) and [*git*](https://git-scm.com/) should be installed via conda to get **MVT** and its requirements.

```bash
conda create -n mvt
conda activate mvt
conda install git pip
```

Then, **MVT** can be downloaded with *git*. Use *pip* to install the required packages from [requirements.txt](requirements.txt).

```bash
git clone https://github.com/pnuehrenberg/multiviewtracks.git
cd mvt
pip install -r requirements.txt
```

Assuming everything worked well, you can now run a jupyter notebook by executing the following commands.

```bash
conda activate mvt # not necessary if still active
jupyter notebook
```

You can import **mvt** by adding the the following lines to your jupyter notebook.

```python
import sys
sys.path.append('path/to/cloned/multiviewtracks') # edit this

import MultiViewTracks as mvt
```

### Usage

We use the [COLMAP](https://colmap.github.io), a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline to reconstruct camera paths and orientations from videos. This is necessary when using a moving camera setup for triangulating animal positions in 3D from multiple-view trajectories. We found COLMAP to be fit for this task, as it is well-documented, open-source and easily-accessible.

Please refer to the example notebooks for [general usage](examples/scene.ipynb) and [visualization](examples/visualization.ipynb) using our python module. You can download the example dataset from [DataShare](https://datashare.mpcdf.mpg.de/s/WBi3T5Oh8QGjOQb).

### License

See the [LICENSE](LICENSE) file for license rights and limitations (MIT).
