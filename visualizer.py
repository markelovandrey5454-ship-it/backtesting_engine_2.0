import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

class PortfolioVisualizer:
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    def plot_scenario_comparison(self, all_strategies_results: dict, scenario_name: str):
        plt.figure(figsize=(14, 7))
        inflation_plotted = False

        for strat_name, df_res in all_strategies_results.items():
            if df_res.empty:
                continue

            # Рисуем кривую номинального капитала этой стратегии
            plt.plot(df_res.index, df_res['Nominal_Capital'], label=f"Модель: {strat_name}", linewidth=2)

            # Рисуем инфляционный эталон (берём из первой попавшейся стратегии)
            if not inflation_plotted:
                plt.plot(df_res.index, df_res['Inflation_Benchmark'],
                         label="Инфляционный Бенчмарк (Цель)", color='black', linestyle='--', linewidth=2.5)
                inflation_plotted = True

        plt.title(f"СРАВНИТЕЛЬНОЕ СОРЕВНОВАНИЕ МОДЕЛЕЙ ОПТИМИЗАЦИИ — СЦЕНАРИЙ {scenario_name.upper()}", fontsize=14, fontweight='bold')
        plt.xlabel("Дата", fontsize=12)
        plt.ylabel("Капитал (руб)", fontsize=12)

        # ИСПРАВЛЕНИЕ: Безопасное и стабильное банковское форматирование оси Y через официальный форматер Matplotlib
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

        plt.legend(loc="upper left", fontsize=11, frameon=True)
        plt.tight_layout()

        file_path = os.path.join(self.output_dir, f"comparison_{scenario_name.lower()}.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"График сценария {scenario_name} успешно сохранен по пути: {file_path}")