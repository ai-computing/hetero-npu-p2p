"""Experimental, model-independent same-host NPU transport contracts."""
from .core import SourceAllocation, DestinationAllocation, CopySpan, TransferPlan, CopySession
from .backend import RngdPdmaBackend
__version__ = '0.4.0'
