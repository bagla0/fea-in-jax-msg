
import numpy as np

def generate_msh_from_sc(input_filepath, output_filepath):
    # 1. Read all non-empty lines from the file
    with open(input_filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 2. Find the metadata line
    meta_idx = -1
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 5:
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
        
        # ==========================================
        # 1. NODES SECTION
        # ==========================================
        f.write("$Nodes\n")
        f.write(f"{n_nodes}\n")
        
        # Split logic entirely by dimension to remove inner 'if' checks
        if dim == 1:
            for line in node_lines:
                parts = line.split()
                # Only parse x. Hardcode y and z to 0.0
                f.write(f"{parts[0]} {float(parts[1]):.8f} 0.00000000 0.00000000\n")
                
        elif dim == 2:
            for line in node_lines:
                parts = line.split()
                # Parse x, y. Hardcode z to 0.0
                f.write(f"{parts[0]} {float(parts[1]):.8f} {float(parts[2]):.8f} 0.00000000\n")
                
        elif dim == 3:
            for line in node_lines:
                parts = line.split()
                # Parse x, y, z directly
                f.write(f"{parts[0]} {float(parts[1]):.8f} {float(parts[2]):.8f} {float(parts[3]):.8f}\n")
                
        f.write("$EndNodes\n")

        # ==========================================
        # 2. ELEMENTS SECTION
        # ==========================================
        f.write("$Elements\n")
        f.write(f"{n_elems}\n")
        
        if dim == 1:
            for line in elem_lines:
                parts = line.split()
                elem_id, mat_id = parts[0], parts[1] 
                
                connectivity = [n for n in parts[2:] if n != '0']
                num_nodes = len(connectivity)
                
                if num_nodes == 2: elem_type = 1
                elif num_nodes == 3: elem_type = 8
                elif num_nodes == 4: elem_type = 26
                elif num_nodes == 5: elem_type = 27  # Correctly maps to 27
                else: elem_type = 1 
                    
                f.write(f"{elem_id} {elem_type} 2 {mat_id} {mat_id} {' '.join(connectivity)}\n")

        elif dim == 2:
            for line in elem_lines:
                parts = line.split()
                elem_id, mat_id = parts[0], parts[1]
                
                connectivity = [n for n in parts[2:] if n != '0']
                num_nodes = len(connectivity)
                
                # Check 2D types first, then catch 1D boundary edges
                if num_nodes == 3: elem_type = 2
                elif num_nodes == 4: elem_type = 3
                elif num_nodes == 5: elem_type = 27  # <-- FIX: Catches 1D 4th-order edges in 2D mesh
                elif num_nodes == 2: elem_type = 1   # <-- FIX: Catches 1D linear edges in 2D mesh
                else: elem_type = 2 
                    
                f.write(f"{elem_id} {elem_type} 2 {mat_id} {mat_id} {' '.join(connectivity)}\n")

        elif dim == 3:
            for line in elem_lines:
                parts = line.split()
                elem_id, mat_id = parts[0], parts[1]
                
                connectivity = [n for n in parts[2:] if n != '0']
                num_nodes = len(connectivity)
                
                # Check 3D types first, then catch lower-dimensional boundary elements
                if num_nodes == 4: elem_type = 4
                elif num_nodes == 10: elem_type = 11
                elif num_nodes == 5: elem_type = 27  # <-- FIX: Catches 1D 4th-order edges in 3D mesh
                # Note: 5 nodes in 3D could technically also be a Pyramid (Type 7). 
                # If you use pyramids, you'll need logic to differentiate them from 4th-order lines.
                else: elem_type = 4 
                    
                f.write(f"{elem_id} {elem_type} 2 {mat_id} {mat_id} {' '.join(connectivity)}\n")
                
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