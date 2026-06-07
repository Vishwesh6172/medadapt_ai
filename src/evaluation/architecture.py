from graphviz import Digraph
import os


def generate_architecture_diagram(run_folder):
    print("\n[Architecture] Generating pipeline diagram...")

    dot = Digraph(format='png')
    dot.attr(rankdir='TB', size='10,12')

    # ==============================
    # NODES
    # ==============================

    dot.node('A', 'Raw Dataset\n(PIMA)', shape='box')
    dot.node('B', 'Cleaning', shape='box')
    dot.node('C', 'Encoding', shape='box')

    dot.node('D', 'Controller\nDecision Engine\n(SMOTE + GAN)', shape='diamond')

    dot.node('E', 'Train-Test Split', shape='box')
    dot.node('F', 'Scaling', shape='box')

    dot.node('G', 'SMOTE\n(Balancing)', shape='box')
    dot.node('H', 'SHAP\nFeature Selection', shape='box')

    dot.node('I', 'Cross Validation\n(XGBoost)', shape='box')
    dot.node('J', 'GAN\n(Data Augmentation)', shape='box')

    dot.node('K', 'Final Training\n(XGBoost)', shape='box')

    dot.node('L', 'Evaluation', shape='box')

    dot.node('M', 'Confusion Matrix', shape='ellipse')
    dot.node('N', 'ROC Curve + AUC', shape='ellipse')
    dot.node('O', 'SHAP Importance', shape='ellipse')

    # ==============================
    # FLOW
    # ==============================

    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')

    dot.edge('E', 'F')
    dot.edge('F', 'G')
    dot.edge('G', 'H')
    dot.edge('H', 'I')

    dot.edge('I', 'J')
    dot.edge('J', 'K')

    dot.edge('K', 'L')

    dot.edge('L', 'M')
    dot.edge('L', 'N')
    dot.edge('L', 'O')

    # ==============================
    # SAVE
    # ==============================

    output_path = os.path.join(run_folder, "architecture")
    dot.render(output_path, cleanup=True)

    print(f"[Architecture] Saved at: {output_path}.png")