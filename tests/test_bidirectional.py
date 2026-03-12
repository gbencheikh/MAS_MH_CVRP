import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


import os
sys.path.append(str(Path(__file__).parent.parent))


from src.instance import VRPInstance
from src.solution import VRPSolution

from src.metaheuristics.hill_climbing import Hill_Climbing
from src.metaheuristics.tabu_search import Tabu_Search
from src.metaheuristics.simulated_annealing import Simulated_Annealing
from src.metaheuristics.genetic_algorithm import Genetic_Algorithm
from src.metaheuristics.ant_colony_am import Ant_Colony_Optimization, Ant
from src.metaheuristics.golden_ball import Golden_Ball
from src.metaheuristics.particle_swarm import Particle_Swarm_Optimization
from src.notifier import *
from src.MAS.bidirectional_communication import BidirectionalMultiAgentSystem


def collect_visit_statistics(solution):
    """
    Collecte les statistiques de visite d'une solution.
    
    Returns:
        Dict avec total_visites, num_unique, num_redundant, max_visits, mean_visits
    """
    if not solution or not hasattr(solution, 'visit_counter'):
        return {
            'total_visites': 0,
            'num_unique': 0,
            'num_redundant': 0,
            'max_visits': 0,
            'mean_visits': 0
        }
    
    vc = solution.visit_counter
    if not vc:
        return {
            'total_visites': 0,
            'num_unique': 0,
            'num_redundant': 0,
            'max_visits': 0,
            'mean_visits': 0
        }
    
    total_visits = sum(vc.values())
    num_unique = len(vc)
    num_redundant = total_visits - num_unique  # Visites en double
    max_visits = max(vc.values())
    mean_visits = total_visits / num_unique if num_unique > 0 else 0
    
    return {
        'total_visites': total_visits,
        'num_unique': num_unique,
        'num_redundant': num_redundant,
        'max_visits': max_visits,
        'mean_visits': mean_visits
    }


def run_multiagent_tests():
    """
    Teste le système multi-agent sur toutes les instances et sauvegarde dans Excel.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(current_dir, "..", "data", "instances")
    results_dir = os.path.join(current_dir, "..", "results")
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"resultats_MAS_{date_str}.xlsx"
    
    os.makedirs(results_dir, exist_ok=True)
    
    Summary_outputfile = os.path.join(results_dir, filename)
    results = []
    
    for file_name in os.listdir(directory_path):
        if not file_name.endswith(".vrp"):
            continue
        
        filepath = os.path.join(directory_path, file_name)
        
        print(f"\n{'='*80}")
        print(f"Instance: {file_name}")
        print(f"{'='*80}")
        
        # Charger l'instance
        instance = VRPInstance.load_from_file(filepath)
        
        # Initialiser la ligne de résultats
        row = {"Instance": file_name}
        if instance.optimal_value is not None:
            row["Optimal"] = instance.optimal_value
        else:
            row["Optimal"] = "None"
        
        # Créer le système multi-agent
        mas = BidirectionalMultiAgentSystem(instance, pool_size=20)
        
        # ============================================================
        # PHASE 1: Agents Itératifs
        # ============================================================
        
        mas.add_iterative_agent(
            name="HC",
            solver=Hill_Climbing,
            run_params={'max_iterations': 1000, 'log_history': True}
        )
        
        mas.add_iterative_agent(
            name="TS",
            solver=Tabu_Search,
            solver_params={'tabu_tenure': 20},
            run_params={'max_iterations': 500, 'aspiration': True, 'log_history': True}
        )
        
        mas.add_iterative_agent(
            name="SA",
            solver=Simulated_Annealing,
            solver_params={
                'initial_temperature': 100.0,
                'cooling_rate': 0.995,
                'min_temperature': 0.0001
            },
            run_params={'max_iterations': 5000, 'seed': 42, 'log_history': True}
        )
        
        # ============================================================
        # PHASE 2: Agents à Population
        # ============================================================
        
        mas.add_population_agent(
            name="GA",
            solver=Genetic_Algorithm,
            solver_params={
                'population_size': max(20, instance.num_customers // 2),
                'elite_size': 5,
                'mutation_rate': 0.4,
                'crossover_rate': 0.9
            },
            run_params={'max_generations': 100, 'seed': 42, 'log_history': True}
        )
        
        mas.add_population_agent(
            name="ACO",
            solver=Ant_Colony_Optimization,
            solver_params={
                'num_ants': 100,
                'alpha': 1.0,
                'beta': 5.0,
                'evaporation_rate': 0.1,
                'q': 100.0
            },
            run_params={'max_iterations': 100, 'local_search': True, 'seed': 42, 'log_history': True}
        )
        
        mas.add_population_agent(
            name="PSO",
            solver=Particle_Swarm_Optimization,
            solver_params={'num_particles': 30, 'w': 0.5, 'c1': 1.5, 'c2': 1.5},
            run_params={'max_iterations': 100, 'seed': 42, 'log_history': True}
        )
        
        mas.add_population_agent(
            name="GB",
            solver=Golden_Ball,
            solver_params={
                'population_size': max(20, instance.num_customers // 3),
                'num_rounds': 5,
                'cooperation_rate': 0.7,
                'mutation_rate': 0.3
            },
            run_params={'max_iterations': 100, 'seed': 42, 'log_history': True}
        )
        
        # ============================================================
        # EXÉCUTION
        # ============================================================
        
        mas_results = mas.run_sequential(verbose=False)
        
        # ============================================================
        # COLLECTE DES RÉSULTATS
        # ============================================================
        
        # Initialiser les compteurs MAS totaux
        mas_total_visits = 0
        mas_all_signatures = {}  # Pour compter les solutions uniques globalement
        
        # Résultats individuels de chaque agent
        all_agent_results = {**mas_results.get('phase1', {}), **mas_results.get('phase2', {})}
        
        for agent_name, agent_data in all_agent_results.items():
            # Distance et CPU
            row[f"{agent_name}_Distance"] = agent_data['distance']
            row[f"{agent_name}_CPU"] = agent_data['time']
            
            # Gap
            if instance.optimal_value is not None:
                row[f"{agent_name}_gap"] = agent_data['distance'] - instance.optimal_value
                row[f"{agent_name}_gap_pct"] = ((agent_data['distance'] - instance.optimal_value) / instance.optimal_value * 100)
            else:
                row[f"{agent_name}_gap"] = "None"
                row[f"{agent_name}_gap_pct"] = "None"
            
            # Faisabilité
            row[f"{agent_name}_Feasible"] = agent_data.get('feasible', False)
            
            # Statistiques de visite complètes
            solution = agent_data.get('solution')
            visit_stats = collect_visit_statistics(solution)
            row[f"{agent_name}_total_visites"] = visit_stats['total_visites']
            row[f"{agent_name}_num_unique"] = visit_stats['num_unique']
            row[f"{agent_name}_num_redundant"] = visit_stats['num_redundant']
            row[f"{agent_name}_max_visits"] = visit_stats['max_visits']
            row[f"{agent_name}_mean_visits"] = visit_stats['mean_visits']
            
            # ✅ AJOUT: Agréger pour les statistiques MAS globales
            mas_total_visits += visit_stats['total_visites']
            
            # Agréger les signatures uniques de tous les agents
            if solution and hasattr(solution, 'visit_counter'):
                for sig, count in solution.visit_counter.items():
                    mas_all_signatures[sig] = mas_all_signatures.get(sig, 0) + count
        
        # Calculer les statistiques MAS globales (tous agents combinés)
        row["MAS_total_visites_global"] = mas_total_visits
        row["MAS_num_unique_global"] = len(mas_all_signatures)
        row["MAS_num_redundant_global"] = mas_total_visits - len(mas_all_signatures)
        if len(mas_all_signatures) > 0:
            row["MAS_max_visits_global"] = max(mas_all_signatures.values())
            row["MAS_mean_visits_global"] = mas_total_visits / len(mas_all_signatures)
        else:
            row["MAS_max_visits_global"] = 0
            row["MAS_mean_visits_global"] = 0
        
        # Résultats globaux du MAS (meilleure solution)
        row["MAS_Distance"] = mas_results['best_distance']
        row["MAS_CPU"] = mas_results['total_time']
        
        if instance.optimal_value is not None:
            row["MAS_gap"] = mas_results['best_distance'] - instance.optimal_value
            row["MAS_gap_pct"] = ((mas_results['best_distance'] - instance.optimal_value) / instance.optimal_value * 100)
        else:
            row["MAS_gap"] = "None"
            row["MAS_gap_pct"] = "None"
        
        # Vérifier faisabilité de la meilleure solution
        if mas_results['best_solution']:
            feasible, msg = mas_results['best_solution'].is_feasible()
            row["MAS_Feasible"] = feasible
        
        # Statistiques du pool
        pool_stats = mas_results.get('pool_stats', {})
        row["Pool_Best"] = pool_stats.get('best', 0)
        row["Pool_Avg"] = pool_stats.get('avg', 0)
        row["Pool_Worst"] = pool_stats.get('worst', 0)
        row["Pool_Count"] = pool_stats.get('count', 0)
        
        # Ajouter la ligne aux résultats
        results.append(row)
        
        # Afficher résumé
        print(f"\n{file_name} - Résultats:")
        print(f"  MAS Distance: {row['MAS_Distance']:.2f}")
        print(f"  MAS CPU: {row['MAS_CPU']:.2f}s")
        print(f"  MAS Visites GLOBALES: {row['MAS_total_visites_global']} (Unique: {row['MAS_num_unique_global']}, Redondant: {row['MAS_num_redundant_global']})")
        print(f"  Pool: Best={row['Pool_Best']:.2f}, Count={row['Pool_Count']}")
        
        # Nettoyage mémoire
        gc.collect()
    
    # ============================================================
    # SAUVEGARDE DANS EXCEL
    # ============================================================
    
    df_summary = pd.DataFrame(results)
    df_summary.to_excel(Summary_outputfile, index=False)
    
    print(f"\n{'='*80}")
    print(f"Résultats sauvegardés dans {Summary_outputfile}")
    print(f"{'='*80}")
    
    return df_summary, results_dir, date_str


def generate_plots(df, results_dir, date_str):
    """
    Génère tous les graphiques de comparaison.
    """
    section("GÉNÉRATION DES GRAPHIQUES")
    
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(results_dir, f"graphics_{date_str}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Filtrer les instances avec valeur optimale
    df_with_opt = df[df['Optimal'] != 'None'].copy()
    df_with_opt['Optimal'] = pd.to_numeric(df_with_opt['Optimal'])
    
    # Liste des agents
    agents = ['HC', 'TS', 'SA', 'GA', 'ACO', 'PSO', 'GB', 'MAS']
    
    # Palette de couleurs
    colors = plt.cm.Set3(np.linspace(0, 1, len(agents)))
    
    # ============================================================
    # GRAPHIQUE 1: Comparaison des distances
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df))
    width = 0.1
    
    for i, agent in enumerate(agents):
        col_name = f"{agent}_Distance"
        if col_name in df.columns:
            distances = df[col_name].values
            ax.bar(x + i*width, distances, width, label=agent, color=colors[i])
    
    ax.set_xlabel('Instances', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Distances par Agent', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * len(agents) / 2)
    ax.set_xticklabels(df['Instance'], rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'distances_{date_str}.png'), dpi=300)
    plt.close()
    print("✓ Graphique des distances sauvegardé")
    
    # ============================================================
    # GRAPHIQUE 2: Comparaison des gaps (%)
    # ============================================================
    
    if not df_with_opt.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(df_with_opt))
        width = 0.1
        
        for i, agent in enumerate(agents):
            col_name = f"{agent}_gap_pct"
            if col_name in df_with_opt.columns:
                gaps = pd.to_numeric(df_with_opt[col_name], errors='coerce').values
                ax.bar(x + i*width, gaps, width, label=agent, color=colors[i])
        
        ax.set_xlabel('Instances', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gap (%)', fontsize=12, fontweight='bold')
        ax.set_title('Comparaison des Gaps par rapport à l\'Optimal (%)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * len(agents) / 2)
        ax.set_xticklabels(df_with_opt['Instance'], rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Optimal')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'gaps_{date_str}.png'), dpi=300)
        plt.close()
        print("✓ Graphique des gaps sauvegardé")
    
    # ============================================================
    # GRAPHIQUE 3: Temps de calcul (CPU)
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df))
    width = 0.1
    
    for i, agent in enumerate(agents):
        col_name = f"{agent}_CPU"
        if col_name in df.columns:
            cpu_times = df[col_name].values
            ax.bar(x + i*width, cpu_times, width, label=agent, color=colors[i])
    
    ax.set_xlabel('Instances', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temps CPU (secondes)', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Temps de Calcul', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * len(agents) / 2)
    ax.set_xticklabels(df['Instance'], rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'cpu_times_{date_str}.png'), dpi=300)
    plt.close()
    print("✓ Graphique des temps CPU sauvegardé")
    
    # ============================================================
    # GRAPHIQUE 4: Statistiques de visite - Total (MAS Global)
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Pour MAS, utiliser les statistiques globales
    x_labels = df['Instance'].values
    x = np.arange(len(x_labels))
    width = 0.35
    
    # Barres pour total visites globales du MAS
    mas_total = df['MAS_total_visites_global'].values
    bar1 = ax.bar(x - width/2, mas_total, width, label='MAS Total Visites', color='steelblue', alpha=0.8)
    
    # Barres pour solutions uniques globales du MAS
    mas_unique = df['MAS_num_unique_global'].values
    bar2 = ax.bar(x + width/2, mas_unique, width, label='MAS Solutions Uniques', color='lightcoral', alpha=0.8)
    
    ax.set_xlabel('Instances', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre de solutions', fontsize=12, fontweight='bold')
    ax.set_title('MAS - Total Visites vs Solutions Uniques (Tous Agents Combinés)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter des valeurs au-dessus des barres
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'mas_global_visits_{date_str}.png'), dpi=300)
    plt.close()
    print("✓ Graphique des visites globales MAS sauvegardé")
    
    # ============================================================
    # GRAPHIQUE 5: Comparaison visites par agent (stacked bar)
    # ============================================================
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df))
    agents_for_visits = ['HC', 'TS', 'SA', 'GA', 'ACO', 'PSO', 'GB']  # Sans MAS
    
    # Créer un stacked bar chart
    bottom = np.zeros(len(df))
    
    for i, agent in enumerate(agents_for_visits):
        col_name = f"{agent}_total_visites"
        if col_name in df.columns:
            visits = df[col_name].values
            ax.bar(x, visits, label=agent, bottom=bottom, color=colors[i])
            bottom += visits
    
    # Ajouter une ligne pour le total MAS global
    mas_total = df['MAS_total_visites_global'].values
    ax.plot(x, mas_total, 'ro-', linewidth=2, markersize=8, label='MAS Total Global', zorder=10)
    
    ax.set_xlabel('Instances', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre de solutions visitées', fontsize=12, fontweight='bold')
    ax.set_title('Visites Cumulées par Agent vs Total MAS', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Instance'], rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'visits_by_agent_stacked_{date_str}.png'), dpi=300)
    plt.close()
    print("✓ Graphique des visites par agent (stacked) sauvegardé")
    # ============================================================
    # GRAPHIQUE 6: Solutions uniques vs redondantes (MAS Global)
    # ============================================================
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    x = np.arange(len(df))
    
    # Sous-graphique 1: Comparaison MAS global
    width = 0.35
    unique_global = df['MAS_num_unique_global'].values
    redundant_global = df['MAS_num_redundant_global'].values
    
    ax1.bar(x - width/2, unique_global, width, label='Solutions Uniques', color='lightgreen', alpha=0.8)
    ax1.bar(x + width/2, redundant_global, width, label='Visites Redondantes', color='salmon', alpha=0.8)
    
    ax1.set_xlabel('Instances', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Nombre de solutions', fontsize=12, fontweight='bold')
    ax1.set_title('MAS Global - Unique vs Redondant', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Instance'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Sous-graphique 2: Taux de redondance (%)
    total_global = df['MAS_total_visites_global'].values
    redundancy_rate = np.divide(redundant_global, total_global, out=np.zeros_like(redundant_global, dtype=float), where=total_global!=0) * 100
    
    bars = ax2.bar(x, redundancy_rate, color='coral', alpha=0.8)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% redondance')
    
    ax2.set_xlabel('Instances', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Taux de redondance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('MAS Global - Taux de Redondance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Instance'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs au-dessus des barres
    for bar, rate in zip(bars, redundancy_rate):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'mas_unique_vs_redundant_{date_str}.png'), dpi=300)
    plt.close()
    print("✓ Graphique unique vs redondant MAS sauvegardé")
    
    # ============================================================
    # GRAPHIQUE 6: Box plot des gaps
    # ============================================================
    
    if not df_with_opt.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        gap_data = []
        labels = []
        
        for agent in agents:
            col_name = f"{agent}_gap_pct"
            if col_name in df_with_opt.columns:
                gaps = pd.to_numeric(df_with_opt[col_name], errors='coerce').dropna()
                if len(gaps) > 0:
                    gap_data.append(gaps)
                    labels.append(agent)
        
        bp = ax.boxplot(gap_data, labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Agents', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gap (%)', fontsize=12, fontweight='bold')
        ax.set_title('Distribution des Gaps par Agent', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'boxplot_gaps_{date_str}.png'), dpi=300)
        plt.close()
        print("✓ Box plot des gaps sauvegardé")
    
    # ============================================================
    # GRAPHIQUE 7: Performance moyenne (radar chart)
    # ============================================================
    
    if not df_with_opt.empty:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Critères de performance
        criteria = ['Gap moyen (%)', 'CPU moyen (s)', 'Visites totales', 'Solutions uniques']
        num_criteria = len(criteria)
        angles = np.linspace(0, 2 * np.pi, num_criteria, endpoint=False).tolist()
        angles += angles[:1]
        
        for agent, color in zip(agents, colors):
            # Calculer les moyennes (normaliser pour le radar)
            gap_col = f"{agent}_gap_pct"
            cpu_col = f"{agent}_CPU"
            visits_col = f"{agent}_total_visites"
            unique_col = f"{agent}_num_unique"
            
            if all(col in df.columns for col in [gap_col, cpu_col, visits_col, unique_col]):
                gap_mean = pd.to_numeric(df_with_opt[gap_col], errors='coerce').mean()
                cpu_mean = df[cpu_col].mean()
                visits_mean = df[visits_col].mean()
                unique_mean = df[unique_col].mean()
                
                # Normaliser (inverser pour gap et cpu : plus petit = mieux)
                values = [
                    max(0, 100 - gap_mean),  # Inverser le gap
                    max(0, 100 - cpu_mean),  # Inverser le CPU
                    visits_mean / 10,        # Échelle
                    unique_mean / 10         # Échelle
                ]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=agent, color=color)
                ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria)
        ax.set_ylim(0, 100)
        ax.set_title('Performance Moyenne Multi-Critères', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'radar_chart_{date_str}.png'), dpi=300)
        plt.close()
        print("✓ Radar chart sauvegardé")
    
    print(f"\n{'='*80}")
    print("TOUS LES GRAPHIQUES ONT ÉTÉ GÉNÉRÉS")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Exécuter les tests
    df, results_dir, date_str = run_multiagent_tests()
    
    # Générer les graphiques
    generate_plots(df, results_dir, date_str)
    
    # Afficher résumé statistique
    print(f"\n{'='*80}")
    print("RÉSUMÉ STATISTIQUE")
    print(f"{'='*80}")
    
    df_with_opt = df[df['Optimal'] != 'None'].copy()
    if not df_with_opt.empty:
        df_with_opt['Optimal'] = pd.to_numeric(df_with_opt['Optimal'])
        
        print("\nGAPS MOYENS (%):")
        agents = ['HC', 'TS', 'SA', 'GA', 'ACO', 'PSO', 'GB', 'MAS']
        for agent in agents:
            col_name = f"{agent}_gap_pct"
            if col_name in df_with_opt.columns:
                gap_mean = pd.to_numeric(df_with_opt[col_name], errors='coerce').mean()
                print(f"  {agent:6s}: {gap_mean:6.2f}%")
        
        print("\nTEMPS CPU MOYENS (s):")
        for agent in agents:
            col_name = f"{agent}_CPU"
            if col_name in df.columns:
                cpu_mean = df[col_name].mean()
                print(f"  {agent:6s}: {cpu_mean:6.2f}s")
    
    print(f"\n{'='*80}")
    print("FIN DE L'ANALYSE")
    print(f"{'='*80}")