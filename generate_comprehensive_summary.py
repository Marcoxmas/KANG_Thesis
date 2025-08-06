#!/usr/bin/env python3

import json
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_all_results():
    """Load all multi-seed test results from JSON files."""
    results_dir = "experiments/multi_seed_results"
    all_results = {}
    
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    print(f"Found {len(json_files)} result files")
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        # Parse filename: {task}_{dataset}_{target}_{seeds}seeds.json
        parts = filename.replace('.json', '').split('_')
        
        if len(parts) >= 3:
            task = parts[0]
            dataset = parts[1]
            if 'seeds' in parts[-1]:
                # Target column is everything between dataset and seeds
                target = '_'.join(parts[2:-1]) if len(parts) > 3 else None
                seeds = parts[-1]
            else:
                target = None
                seeds = parts[-1]
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                key = f"{task}_{dataset}"
                if target:
                    key += f"_{target}"
                
                all_results[key] = {
                    'task': task,
                    'dataset': dataset,
                    'target': target,
                    'data': data,
                    'file': filename
                }
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return all_results

def calculate_statistics(results):
    """Calculate comprehensive statistics for each test."""
    stats_data = []
    
    for key, result in results.items():
        task = result['task']
        dataset = result['dataset']
        target = result['target']
        data = result['data']
        
        for config_name, seed_results in data.items():
            valid_results = [r['result'] for r in seed_results if r['result'] is not None]
            
            if valid_results:
                stats_entry = {
                    'test_name': key,
                    'task': task,
                    'dataset': dataset,
                    'target': target or 'N/A',
                    'config': config_name,
                    'n_seeds': len(valid_results),
                    'mean': np.mean(valid_results),
                    'std': np.std(valid_results),
                    'min': np.min(valid_results),
                    'max': np.max(valid_results),
                    'median': np.median(valid_results),
                    'cv': np.std(valid_results) / np.mean(valid_results) if np.mean(valid_results) != 0 else 0
                }
                
                # Calculate confidence interval
                if len(valid_results) > 1:
                    confidence = 0.95
                    n = len(valid_results)
                    se = stats_entry['std'] / np.sqrt(n)
                    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
                    stats_entry['ci_lower'] = stats_entry['mean'] - h
                    stats_entry['ci_upper'] = stats_entry['mean'] + h
                else:
                    stats_entry['ci_lower'] = stats_entry['mean']
                    stats_entry['ci_upper'] = stats_entry['mean']
                
                stats_data.append(stats_entry)
    
    return pd.DataFrame(stats_data)

def analyze_global_features_impact(results):
    """Analyze the impact of global features across all tests."""
    comparisons = []
    
    for key, result in results.items():
        data = result['data']
        
        without_results = [r['result'] for r in data.get('without_global_features', []) if r['result'] is not None]
        with_results = [r['result'] for r in data.get('with_global_features', []) if r['result'] is not None]
        
        if without_results and with_results:
            task = result['task']
            
            without_mean = np.mean(without_results)
            with_mean = np.mean(with_results)
            
            if task == "regression":
                # For regression (MAE), lower is better
                improvement = ((without_mean - with_mean) / without_mean) * 100
                better_config = "with_global" if with_mean < without_mean else "without_global"
            else:
                # For classification (ROC-AUC/Accuracy), higher is better
                improvement = ((with_mean - without_mean) / without_mean) * 100
                better_config = "with_global" if with_mean > without_mean else "without_global"
            
            # Paired t-test
            try:
                t_stat, p_value = stats.ttest_rel(without_results, with_results)
                significant = p_value < 0.05
                
                # Cohen's d for effect size
                differences = np.array(with_results) - np.array(without_results)
                cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0
            except:
                t_stat, p_value, significant, cohens_d = np.nan, np.nan, False, np.nan
            
            comparisons.append({
                'test_name': key,
                'task': result['task'],
                'dataset': result['dataset'],
                'target': result['target'] or 'N/A',
                'without_mean': without_mean,
                'with_mean': with_mean,
                'improvement_pct': improvement,
                'better_config': better_config,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': significant,
                'cohens_d': cohens_d,
                'n_seeds': len(without_results)
            })
    
    return pd.DataFrame(comparisons)

def create_toxcast_descriptions():
    """Create descriptions for the ToxCast assays used in the thesis."""
    descriptions = {
        'TOX21_AhR_LUC_Agonist': 'Measures the activation of the Aryl hydrocarbon Receptor pathway',
        'TOX21_Aromatase_Inhibition': 'Assesses inhibition of the aromatase enzyme, important in hormone biosynthesis', 
        'TOX21_AutoFluor_HEK293_Cell_blue': 'Control task to detect autofluorescence in HEK293 cells under blue channel excitation',
        'TOX21_p53_BLA_p3_ch1': 'Measures activation of the tumor suppressor protein p53 using a reporter assay',
        'TOX21_p53_BLA_p4_ratio': 'Additional reporter-based p53 activation task capturing response ratios'
    }
    return descriptions

def generate_visualizations(stats_df, comparisons_df, output_dir):
    """Generate comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance comparison by dataset
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Regression tasks
    reg_data = stats_df[stats_df['task'] == 'regression']
    if not reg_data.empty:
        # Without global features
        reg_without = reg_data[reg_data['config'] == 'without_global_features']
        if not reg_without.empty:
            x_pos = range(len(reg_without))
            axes[0, 0].bar(x_pos, reg_without['mean'], 
                          yerr=reg_without['std'], capsize=5, alpha=0.7)
            axes[0, 0].set_title('Regression Performance (Without Global Features)', fontsize=14)
            axes[0, 0].set_ylabel('MAE', fontsize=12)
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels([f"{row['dataset']}\n{row['target']}" for _, row in reg_without.iterrows()], 
                                      rotation=45, ha='right', fontsize=10)
            axes[0, 0].grid(True, alpha=0.3)
        
        # With global features
        reg_with = reg_data[reg_data['config'] == 'with_global_features']
        if not reg_with.empty:
            x_pos = range(len(reg_with))
            axes[0, 1].bar(x_pos, reg_with['mean'], 
                          yerr=reg_with['std'], capsize=5, alpha=0.7, color='orange')
            axes[0, 1].set_title('Regression Performance (With Global Features)', fontsize=14)
            axes[0, 1].set_ylabel('MAE', fontsize=12)
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels([f"{row['dataset']}\n{row['target']}" for _, row in reg_with.iterrows()], 
                                      rotation=45, ha='right', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
    
    # Classification tasks
    cls_data = stats_df[stats_df['task'] == 'classification']
    if not cls_data.empty:
        # Without global features
        cls_without = cls_data[cls_data['config'] == 'without_global_features']
        if not cls_without.empty:
            x_pos = range(len(cls_without))
            axes[1, 0].bar(x_pos, cls_without['mean'], 
                          yerr=cls_without['std'], capsize=5, alpha=0.7, color='green')
            axes[1, 0].set_title('Classification Performance (Without Global Features)', fontsize=14)
            axes[1, 0].set_ylabel('ROC-AUC', fontsize=12)
            axes[1, 0].set_xticks(x_pos)
            # Use cleaner labels for ToxCast
            labels = []
            for _, row in cls_without.iterrows():
                if row['dataset'].startswith('TOX21'):
                    labels.append(row['dataset'].replace('TOX21_', '').replace('_', '\n'))
                else:
                    labels.append(f"{row['dataset']}\n{row['target']}")
            axes[1, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
        
        # With global features
        cls_with = cls_data[cls_data['config'] == 'with_global_features']
        if not cls_with.empty:
            x_pos = range(len(cls_with))
            axes[1, 1].bar(x_pos, cls_with['mean'], 
                          yerr=cls_with['std'], capsize=5, alpha=0.7, color='red')
            axes[1, 1].set_title('Classification Performance (With Global Features)', fontsize=14)
            axes[1, 1].set_ylabel('ROC-AUC', fontsize=12)
            axes[1, 1].set_xticks(x_pos)
            # Use cleaner labels for ToxCast
            labels = []
            for _, row in cls_with.iterrows():
                if row['dataset'].startswith('TOX21'):
                    labels.append(row['dataset'].replace('TOX21_', '').replace('_', '\n'))
                else:
                    labels.append(f"{row['dataset']}\n{row['target']}")
            axes[1, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Global features impact
    if not comparisons_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Improvement percentage distribution
        axes[0, 0].hist(comparisons_df['improvement_pct'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='No improvement', linewidth=2)
        axes[0, 0].set_xlabel('Improvement Percentage (%)', fontsize=12)
        axes[0, 0].set_ylabel('Number of Tests', fontsize=12)
        axes[0, 0].set_title('Distribution of Global Features Impact', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Significance vs Effect Size
        significant_mask = comparisons_df['significant']
        axes[0, 1].scatter(comparisons_df.loc[~significant_mask, 'cohens_d'], 
                          comparisons_df.loc[~significant_mask, 'improvement_pct'], 
                          alpha=0.6, label='Not Significant', color='gray', s=60)
        axes[0, 1].scatter(comparisons_df.loc[significant_mask, 'cohens_d'], 
                          comparisons_df.loc[significant_mask, 'improvement_pct'], 
                          alpha=0.8, label='Significant (p<0.05)', color='red', s=60)
        axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].set_xlabel("Cohen's d (Effect Size)", fontsize=12)
        axes[0, 1].set_ylabel('Improvement Percentage (%)', fontsize=12)
        axes[0, 1].set_title('Statistical Significance vs Effect Size', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # P-value distribution
        valid_pvalues = comparisons_df['p_value'].dropna()
        if len(valid_pvalues) > 0:
            axes[1, 0].hist(valid_pvalues, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(0.05, color='red', linestyle='--', label='p=0.05', linewidth=2)
            axes[1, 0].set_xlabel('P-value', fontsize=12)
            axes[1, 0].set_ylabel('Number of Tests', fontsize=12)
            axes[1, 0].set_title('P-value Distribution', fontsize=14)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Task-specific improvement
        task_improvement = comparisons_df.groupby('task')['improvement_pct'].agg(['mean', 'std', 'count'])
        x_pos = range(len(task_improvement))
        axes[1, 1].bar(x_pos, task_improvement['mean'], 
                      yerr=task_improvement['std'], capsize=5, alpha=0.7)
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[1, 1].set_ylabel('Mean Improvement Percentage (%)', fontsize=12)
        axes[1, 1].set_title('Global Features Impact by Task Type', fontsize=14)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(task_improvement.index, fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add count annotations
        for i, (task, row) in enumerate(task_improvement.iterrows()):
            y_pos = row['mean'] + (row['std'] if not pd.isna(row['std']) else 0) + 0.5
            axes[1, 1].text(i, y_pos, f'n={row["count"]}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'global_features_impact.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_reports(stats_df, comparisons_df, results, output_dir):
    """Generate comprehensive text reports."""
    toxcast_descriptions = create_toxcast_descriptions()
    
    # Main summary report
    with open(os.path.join(output_dir, 'comprehensive_summary.txt'), 'w') as f:
        f.write("KANG THESIS - COMPREHENSIVE MULTI-SEED TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total tests performed: {len(results)}\n")
        f.write(f"Total configurations tested: {len(stats_df)}\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        # Overall statistics
        reg_tests = len(stats_df[stats_df['task'] == 'regression'])
        cls_tests = len(stats_df[stats_df['task'] == 'classification'])
        f.write(f"Regression tests: {reg_tests}\n")
        f.write(f"Classification tests: {cls_tests}\n\n")
        
        # Dataset breakdown
        dataset_counts = stats_df.groupby('dataset')['test_name'].nunique()
        f.write("DATASET BREAKDOWN:\n")
        for dataset, count in dataset_counts.items():
            f.write(f"  {dataset}: {count} tests\n")
        f.write("\n")
        
        # Global features impact summary
        if not comparisons_df.empty:
            improvements = comparisons_df['improvement_pct']
            positive_improvements = (improvements > 0).sum()
            significant_improvements = comparisons_df['significant'].sum()
            
            f.write("GLOBAL FEATURES IMPACT SUMMARY:\n")
            f.write(f"  Tests with positive improvement: {positive_improvements}/{len(comparisons_df)} ({positive_improvements/len(comparisons_df)*100:.1f}%)\n")
            f.write(f"  Statistically significant improvements: {significant_improvements}/{len(comparisons_df)} ({significant_improvements/len(comparisons_df)*100:.1f}%)\n")
            f.write(f"  Mean improvement: {improvements.mean():.2f}% (± {improvements.std():.2f}%)\n")
            f.write(f"  Best improvement: {improvements.max():.2f}% ({comparisons_df.loc[improvements.idxmax(), 'test_name']})\n")
            f.write(f"  Worst performance: {improvements.min():.2f}% ({comparisons_df.loc[improvements.idxmin(), 'test_name']})\n\n")
        
        # ToxCast assay descriptions
        f.write("TOXCAST ASSAY DESCRIPTIONS:\n")
        f.write("-" * 40 + "\n")
        for assay, description in toxcast_descriptions.items():
            f.write(f"{assay}:\n")
            f.write(f"  {description}\n\n")
        
        # Detailed results by dataset
        f.write("DETAILED RESULTS BY DATASET\n")
        f.write("-" * 40 + "\n\n")
        
        for dataset in sorted(stats_df['dataset'].unique()):
            dataset_data = stats_df[stats_df['dataset'] == dataset]
            f.write(f"{dataset.upper()} DATASET:\n")
            
            for task in dataset_data['task'].unique():
                task_data = dataset_data[dataset_data['task'] == task]
                f.write(f"  {task.capitalize()} Task:\n")
                
                for target in sorted(task_data['target'].unique()):
                    target_data = task_data[task_data['target'] == target]
                    f.write(f"    Target: {target}\n")
                    
                    # Add description for ToxCast assays
                    if target in toxcast_descriptions:
                        f.write(f"    Description: {toxcast_descriptions[target]}\n")
                    
                    for _, row in target_data.iterrows():
                        f.write(f"      {row['config'].replace('_', ' ').title()}:\n")
                        f.write(f"        Mean ± Std: {row['mean']:.4f} ± {row['std']:.4f}\n")
                        f.write(f"        Range: [{row['min']:.4f}, {row['max']:.4f}]\n")
                        f.write(f"        95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]\n")
                        f.write(f"        CV: {row['cv']:.4f}\n\n")
            
            f.write("\n")
        
        # Statistical comparisons
        if not comparisons_df.empty:
            f.write("STATISTICAL COMPARISONS\n")
            f.write("-" * 40 + "\n\n")
            
            # Sort by improvement percentage
            sorted_comparisons = comparisons_df.sort_values('improvement_pct', ascending=False)
            
            f.write("TOP IMPROVEMENTS:\n")
            for _, row in sorted_comparisons.head(min(10, len(sorted_comparisons))).iterrows():
                f.write(f"  {row['test_name']}: {row['improvement_pct']:+.2f}% ")
                f.write(f"({'significant' if row['significant'] else 'not significant'})\n")
            
            if len(sorted_comparisons) > 10:
                f.write("\nBOTTOM PERFORMANCES:\n")
                for _, row in sorted_comparisons.tail(min(10, len(sorted_comparisons))).iterrows():
                    f.write(f"  {row['test_name']}: {row['improvement_pct']:+.2f}% ")
                    f.write(f"({'significant' if row['significant'] else 'not significant'})\n")
            
            # Task-specific analysis
            f.write(f"\nTASK-SPECIFIC ANALYSIS:\n")
            for task in comparisons_df['task'].unique():
                task_data = comparisons_df[comparisons_df['task'] == task]
                f.write(f"\n{task.upper()} TASKS:\n")
                f.write(f"  Number of tests: {len(task_data)}\n")
                f.write(f"  Mean improvement: {task_data['improvement_pct'].mean():.2f}%\n")
                f.write(f"  Std improvement: {task_data['improvement_pct'].std():.2f}%\n")
                f.write(f"  Significant improvements: {task_data['significant'].sum()}/{len(task_data)}\n")
                
                if len(task_data) > 0:
                    best_idx = task_data['improvement_pct'].idxmax()
                    worst_idx = task_data['improvement_pct'].idxmin()
                    f.write(f"  Best performer: {task_data.loc[best_idx, 'test_name']} ({task_data.loc[best_idx, 'improvement_pct']:+.2f}%)\n")
                    f.write(f"  Worst performer: {task_data.loc[worst_idx, 'test_name']} ({task_data.loc[worst_idx, 'improvement_pct']:+.2f}%)\n")
    
    # CSV exports for further analysis
    stats_df.to_csv(os.path.join(output_dir, 'detailed_statistics.csv'), index=False)
    if not comparisons_df.empty:
        comparisons_df.to_csv(os.path.join(output_dir, 'global_features_comparisons.csv'), index=False)
    
    # Create a summary table for LaTeX
    with open(os.path.join(output_dir, 'latex_summary_table.txt'), 'w') as f:
        f.write("% LaTeX table for thesis results\n")
        f.write("% Copy this into your thesis document\n\n")
        
        if not comparisons_df.empty:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Multi-seed test results comparing performance with and without global features}\n")
            f.write("\\label{tab:multiseed_results}\n")
            f.write("\\begin{tabular}{|l|l|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Dataset} & \\textbf{Target} & \\textbf{Without Global} & \\textbf{With Global} & \\textbf{Improvement} & \\textbf{p-value} \\\\\n")
            f.write("\\hline\n")
            
            for _, row in comparisons_df.iterrows():
                dataset = row['dataset']
                target = row['target'] if row['target'] != 'N/A' else ''
                without_mean = row['without_mean']
                with_mean = row['with_mean']
                improvement = row['improvement_pct']
                p_value = row['p_value']
                
                # Format p-value
                if pd.isna(p_value):
                    p_str = "N/A"
                elif p_value < 0.001:
                    p_str = "< 0.001"
                else:
                    p_str = f"{p_value:.3f}"
                
                # Add significance marker
                significance = "*" if row['significant'] else ""
                
                f.write(f"{dataset} & {target} & {without_mean:.4f} & {with_mean:.4f} & {improvement:+.2f}\\%{significance} & {p_str} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            f.write("% * indicates statistically significant difference (p < 0.05)\n")
    
    print(f"Reports generated in {output_dir}/")

def main():
    print("Loading all multi-seed test results...")
    results = load_all_results()
    
    if not results:
        print("No results found! Make sure multi-seed tests have been run.")
        return
    
    print(f"Loaded {len(results)} test results")
    
    print("Calculating comprehensive statistics...")
    stats_df = calculate_statistics(results)
    
    print("Analyzing global features impact...")
    comparisons_df = analyze_global_features_impact(results)
    
    output_dir = "experiments/multi_seed_summaries"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    generate_visualizations(stats_df, comparisons_df, output_dir)
    
    print("Generating comprehensive reports...")
    generate_reports(stats_df, comparisons_df, results, output_dir)
    
    print("\nSUMMARY GENERATION COMPLETE!")
    print(f"Results saved in: {output_dir}/")
    print("Key files:")
    print(f"  - comprehensive_summary.txt: Main text report")
    print(f"  - detailed_statistics.csv: Raw statistics data")
    print(f"  - global_features_comparisons.csv: Global features impact analysis")
    print(f"  - performance_overview.png: Performance comparison plots")
    print(f"  - global_features_impact.png: Global features impact visualization")
    print(f"  - latex_summary_table.txt: LaTeX table for thesis")

if __name__ == "__main__":
    main()
