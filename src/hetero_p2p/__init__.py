"""Experimental, model-independent same-host NPU transport contracts."""
from .core import SourceAllocation, DestinationAllocation, CopySpan, TransferPlan, CopySession
from .backend import RngdPdmaBackend
from .gpu import CudaRegion, DmabufCopy, RngdDmabufBackend, cuda_region, release_region
from .paths import PathUnavailable, ProprietaryModuleError
__version__ = '0.8.0'

from .serving import BatchOptions, CompletionLedger, RngdBinding
