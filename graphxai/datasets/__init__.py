from .dataset import get_dataset, GraphDataset, NodeDataset
from .load_synthetic import load_ShapeGGen
from .shape_graph import ShapeGGen
from .ba_houses import BAHouses

# Real-world datasets:
from .real_world.MUTAG import MUTAG,DD
from .real_world.benzene.benzene import Benzene
from .real_world.fluoride_carbonyl.fluoride_carbonyl import FluorideCarbonyl
from .real_world.mutagenicity import Mutagenicity
from .real_world.alkane_carbonyl.alkane_carbonyl import AlkaneCarbonyl
from .ba3motif import BA3Motif
from .syn_datasets import BAHouse,BADiamond,BAWheel,BACycle