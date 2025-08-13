from .hopper import HopperEnv
from .swimmer import SwimmerEnv
from .ant import AntEnv
from .gap import GapEnv
from .walker import WalkerEnv
from .ant_wall import AntWallEnv
from .ant_tunnel import AntTunnelEnv
from .ant_gap import AntGapEnv
from .walker_single import WalkerSingleEnv
from .walker_reconfig import WalkerReconfigEnv
from .ant_single_reconfig import AntSingleReconfigEnv
from .ant_tunnel_reconfig import AntTunnelReconfigEnv
from .ant_single import AntSingleEnv
from .walker_neo_reconfig import WalkerNeoReconfigEnv


env_dict = {
    'hopper': HopperEnv,
    'swimmer': SwimmerEnv,
    'ant': AntEnv,
    'gap': GapEnv,
    'walker': WalkerEnv,
    'ant_wall': AntWallEnv,
    'ant_tunnel': AntTunnelEnv,
    'ant_gap': AntGapEnv,
    'walker_single': WalkerSingleEnv,
    'walker_reconfig': WalkerReconfigEnv,
    'ant_single_reconfig': AntSingleReconfigEnv,
    'ant_tunnel_reconfig': AntTunnelReconfigEnv,
    'ant_single': AntSingleEnv,
    'walker_neo_reconfig': WalkerNeoReconfigEnv,
}