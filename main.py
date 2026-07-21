import pandas as pd
import warnings
import logging
import os
from sklearn.exceptions import ConvergenceWarning
from backtest_engine import PortfolioOrchestrator, PortfolioBacktester
from visualizer import PortfolioVisualizer

from benchmarks import (
                        UniformStrategy, RandomMonkeyStrategy, StochasticMomentumStrategy, PersonalProfileStrategy, MarkowitzStrategy, MlHeavyweightStrategy,
                        RobustParabolicCvarStrategy)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

if __name__ == "__main__":
    print("=== ЗАПУСК ГЛОБАЛЬНОГО ИСТОРИЧЕСКОГО БЭКТЕСТА СИСТЕМЫ ===")

    panel_path = 'data/matrix/global_board_panel.csv'
    vol_path = 'data/matrix/global_volatility_panel.csv'

    if not os.path.exists(panel_path) or not os.path.exists(vol_path):
        raise FileNotFoundError("Критические матрицы данных не найдены! Сначала запустите скрипт data_sync.py.")

    board_panel = pd.read_csv(panel_path, parse_dates=['Date']).set_index('Date')
    volatility_panel = pd.read_csv(vol_path, parse_dates=['Date']).set_index('Date')

    strategies = [
        UniformStrategy(), RandomMonkeyStrategy(), StochasticMomentumStrategy(), PersonalProfileStrategy(), MarkowitzStrategy(), MlHeavyweightStrategy(),
        RobustParabolicCvarStrategy()
    ]

    print("\n[ШАГ 1/3] Запуск динамической симуляции ребалансировок моделей...")
    orchestrator = PortfolioOrchestrator(board_panel, strategies)

    orchestrator.volatility_panel = volatility_panel

    all_weights = orchestrator.generate_weights_history(start_date="2015-01-01")

    print("\n[ШАГ 2/3] Расчет накопления капитала по макро-сценариям...")
    backtester = PortfolioBacktester(board_panel, inflation_annual=0.075, commission=0.0005)
    visualizer = PortfolioVisualizer()

    lump_results = {}
    dca_results = {}
    fire_results = {}

    for strat in strategies:
        name = strat.name
        logging.info(f"Прогон капитала для модели: {name}")

        lump_results[name] = backtester.run_lumpsum_simulation(all_weights[name], initial_capital=1000000.0)
        dca_results[name] = backtester.run_dca_simulation(all_weights[name], extra_capital=50000.0)
        fire_results[name] = backtester.run_fire_simulation(all_weights[name], initial_capital=6000000.0, wherewithal=60000.0)

    print("\n[ШАГ 3/3] Генерация и визуализация сравнительных графиков...")
    visualizer.plot_scenario_comparison(lump_results, "lumpsum")
    visualizer.plot_scenario_comparison(dca_results, "dca")
    visualizer.plot_scenario_comparison(fire_results, "fire")

    print("\n=== БЭКТЕСТ УСПЕШНО ЗАВЕРШЕН! ПРОВЕРЯЙ КАРТИНКИ В data/results/ ===")
