import uproot
import pandas as pd

# <path_to_rootfiles> and <tree_name> are str type

root_file = uproot.open(<path_to_rootfiles>)
tree = root_file[<tree_name>]
columns_load = tree.keys()

data_dict = {f'{branch}':tree[branch].array(library="pd") for branch in columns_load}

df = pd.DataFrame(data_dict)
