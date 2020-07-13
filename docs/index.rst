MultiViewTracks
===============

| **mvt** integrates the results from structure-from-motion (SfM) and video-based tracking.
| **mvt** can be used to acquire highly-detailed animal trajectories for behavioral analyses.

-------------

About
=====

You can find can find our paper at `movement ecology
<https://www.biorxiv.org/content/10.1101/571232v1>`_, where we used **mvt** in diverse aquatic environments.

.. raw:: html
   :file: reconstruction.html

--------------

How to
======

Visit the `GitHub repository
<https://github.com/pnuehrenberg/multiviewtracks>`_ for installation instructions.

For examples and reference of the python module, see the following pages.

.. toctree::
   :maxdepth: 2
   :glob:

   examples/*

   ref_scene
   ref_camera
   ref_tracks
   ref_utils

:ref:`genindex`

-------------

References
==========

We use `COLMAP
<https://colmap.github.io>`_ [2016sfm]_, [2016mvs]_, [2016vote]_, a general-purpose Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline to reconstruct camera paths and orientations from videos.
This is necessary when using a moving camera setup for triangulating animal positions in 3D from multiple-view trajectories.
We found COLMAP to be fit for this task, as it is well-documented, open-source and easily-accessible.

.. [2016sfm] Schönberger, J. L., & Frahm, J. M. (2016). Structure-from-motion revisited.
  In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4104-4113).

.. [2016mvs] Schönberger, J. L., Zheng, E., Frahm, J. M., & Pollefeys, M. (2016, October).
    Pixelwise view selection for unstructured multi-view stereo.
    In European Conference on Computer Vision (pp. 501-518). Springer, Cham.

.. [2016vote] Schönberger, J. L., Price, T., Sattler, T., Frahm, J. M., & Pollefeys, M. (2016, November).
    A vote-and-verify strategy for fast spatial verification in image retrieval.
    In Asian Conference on Computer Vision (pp. 321-337). Springer, Cham.
