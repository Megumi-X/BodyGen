from .hopper import HopperEnv
from .swimmer import SwimmerEnv
from .ant import AntEnv
from .gap import GapEnv
from .walker import WalkerEnv
from .ant_wall import AntWallEnv
from .walker_wall import WalkerWallEnv
from .ant_tunnel import AntTunnelEnv
from .ant_gap import AntGapEnv
from .walker_gap import WalkerGapEnv
from .walker_tunnel import WalkerTunnelEnv
from .walker_single import WalkerSingleEnv
from .walker_mount import WalkerMountEnv
from .walker_mount_reconfig import WalkerMountReconfigEnv
from .walker_gap_reconfig import WalkerGapReconfigEnv
from .walker_sand import WalkerSandEnv
from .walker_wall_reconfig import WalkerWallReconfigEnv
from .walker_sand_reconfig import WalkerSandReconfigEnv


env_dict = {
    'hopper': HopperEnv,
    'swimmer': SwimmerEnv,
    'ant': AntEnv,
    'gap': GapEnv,
    'walker': WalkerEnv,
    'ant_wall': AntWallEnv,
    'walker_wall': WalkerWallEnv,
    'ant_tunnel': AntTunnelEnv,
    'ant_gap': AntGapEnv,
    'walker_gap': WalkerGapEnv,
    'walker_tunnel': WalkerTunnelEnv,
    'walker_single': WalkerSingleEnv,
    'walker_mount': WalkerMountEnv,
    'walker_mount_reconfig': WalkerMountReconfigEnv,
    'walker_gap_reconfig': WalkerGapReconfigEnv,
    'walker_sand': WalkerSandEnv,
    'walker_wall_reconfig': WalkerWallReconfigEnv,
    'walker_sand_reconfig': WalkerSandReconfigEnv,
}