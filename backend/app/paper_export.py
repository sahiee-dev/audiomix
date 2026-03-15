import json
import numpy as np
from typing import Dict, List

class PaperDataExporter:
    """
    Export data in formats suitable for research papers
    """
    
    @staticmethod
    def export_latex_table(data: List[Dict], output_path: str):
        """
        Generate LaTeX table for paper
        
        Example output:
        \\begin{table}
        \\begin{tabular}{|l|r|r|r|}
        ...
        """
        with open(output_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{|l|r|r|r|r|}\n")
            f.write("\\hline\n")
            f.write("Track Pair & BPM Match & Energy Match & Pattern Score & Smoothness \\\\\n")
            f.write("\\hline\n")
            
            for row in data:
                f.write(f"{row['pair']} & {row['bpm']:.2f} & {row['energy']:.2f} & "
                       f"{row['score']:.2f} & {row['smoothness']:.2f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Automatic DJ Mixing Results}\n")
            f.write("\\label{tab:results}\n")
            f.write("\\end{table}\n")
        
        print(f"📄 LaTeX table exported: {output_path}")
    
    @staticmethod
    def export_json_dataset(experiments: List[Dict], output_path: str):
        """Export complete dataset in JSON format for reproducibility"""
        dataset = {
            'metadata': {
                'version': '1.0',
                'description': 'Automatic DJ Mixing Experiment Dataset',
                'num_experiments': len(experiments)
            },
            'experiments': experiments
        }
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"📄 JSON dataset exported: {output_path}")
