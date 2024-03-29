from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from dataset.mpii import MPIIDataset as mpii
# from dataset.h36m import H36MDataset as h36m
from .multiview_h36m import MultiViewH36M as multiview_h36m
from .h36m import H36M as h36m
from .mpii3d import MPII3D as mpii3d
# from dataset.mixed_dataset import MixedDataset as mixed
from .multiview_mpii3d import MultiViewMPII3D as multiview_mpii3d
from .multiview_totalcapture import MultiViewTotalCapture as multiview_totalcapture
from .mocap_dataset import MoCapDataset as mocap_dataset