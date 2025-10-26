from treelib import Tree
import os
import yaml

def print_directory_tree_treelib(path):
    """Print directory structure using treelib"""
    tree = Tree()
    
    def add_to_tree(path, parent=None):
        node_id = path
        node_name = os.path.basename(path) or path
        
        tree.create_node(node_name, node_id, parent=parent)
        
        if os.path.isdir(path):
            try:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    add_to_tree(item_path, node_id)
            except PermissionError:
                pass
    
    add_to_tree(path)
    return tree

def load_configs(crew_name: str, configs_path: str = "configs/"):
    configs_details = {
        "agents": os.path.join(configs_path, f"{crew_name}_agents.yaml"),
        "tasks": os.path.join(configs_path, f"{crew_name}_tasks.yaml"),
    }
    configs = {}
    for config_detail, path in configs_details.items():
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            configs[config_detail] = config
    return configs

if __name__ == "__main__":
    # Usage
    print(print_directory_tree_treelib('knowledge'))
    