
import numpy as np

def generate_msh_from_sc(input_filepath, output_filepath):
    # 1. Read all non-empty lines from the file
    with open(input_filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 2. Find the metadata line
    meta_idx = -1
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit() and parts[1].isdigit() and int(parts[1]) > 100:
            meta_idx = i
            dim = int(parts[0])
            n_nodes = int(parts[1])
            n_elems = int(parts[2])
            n_mats = int(parts[3])
            break

    if meta_idx == -1:
        raise ValueError("Could not find the metadata row defining mesh size.")

    # 3. Slice the lists for nodes, elements, and materials
    node_lines = lines[meta_idx + 1 : meta_idx + 1 + n_nodes]
    elem_lines = lines[meta_idx + 1 + n_nodes : meta_idx + 1 + n_nodes + n_elems]

    # 4. Write the .msh file and track cell types
    cell_types_found = set()
    
    with open(output_filepath, 'w') as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        
        # Nodes
        f.write("$Nodes\n")
        f.write(f"{n_nodes}\n")
        for line in node_lines:
            parts = line.split()
            node_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            z = float(parts[3]) if len(parts) > 3 else 0.0
            f.write(f"{node_id} {x:.8f} {y:.8f} {z:.8f}\n")
        f.write("$EndNodes\n")
        
        # Elements
        f.write("$Elements\n")
        f.write(f"{n_elems}\n")
        for line in elem_lines:
            parts = line.split()
            elem_id, mat_id = int(parts[0]), int(parts[1])
            
            connectivity = []
            for node_str in parts[2:]:
                if int(node_str) == 0: break
                connectivity.append(node_str)
                
            num_nodes = len(connectivity)
            
            # Determine Gmsh Element Type & Track for the print summary
            if num_nodes == 3: 
                elem_type = 2
                cell_types_found.add("3-Node Triangle")
            elif num_nodes == 10: 
                elem_type = 11
                cell_types_found.add("10-Node Tetrahedron")
            elif num_nodes == 4: 
                elem_type = 4 if dim == 3 else 3
                cell_types_found.add("4-Node Tetrahedron" if dim == 3 else "4-Node Quad")
            else: 
                elem_type = 2
                cell_types_found.add("Unknown Type")
                
            node_seq = " ".join(connectivity)
            f.write(f"{elem_id} {elem_type} 2 {mat_id} {mat_id} {node_seq}\n")
            
        f.write("$EndElements\n")

    # ==========================================
    # 6. PRINT SUMMARY
    # ==========================================
    print("--- MESH SUMMARY ---\n")
    print(f"SG:          {dim}D")
    print(f"Num Nodes:   {n_nodes}")
    print(f"Num Cells:   {n_elems}")
    print(f"Num Mat:     {n_mats}")
    print(f"Cell Types:  {', '.join(cell_types_found)}")
    return dim