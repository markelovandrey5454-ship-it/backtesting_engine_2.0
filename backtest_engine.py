import pandas as pd
import numpy as np
import logging


class PortfolioOrchestrator:
    def __init__(self, board_panel: pd.DataFrame, strategies: list, commission: float = 0.0005):
        self.board_panel = board_panel.sort_index()
        self.strategies = strategies
        self.commission = commission
        self.asset_tickers = [col for col in self.board_panel.columns if not col.endswith('_div')]

    def generate_weights_history(self, start_date: str = "2015-01-01", rebalance_months: int = 1) -> dict[ str, pd.DataFrame]:
        test_returns = self.board_panel[self.asset_tickers].loc[start_date:]
        sim_dates = test_returns.index

        weights_histories = {
            strat.name: pd.DataFrame(0.0, index=sim_dates, columns=self.asset_tickers)
            for strat in self.strategies
        }

        current_weights = {strat.name: np.zeros(len(self.asset_tickers)) for strat in self.strategies}

        for t_idx, current_date in enumerate(sim_dates):
            for strat in self.strategies:
                name = strat.name

                historical_slice = self.board_panel[self.asset_tickers].loc[:current_date]
                if len(historical_slice) > 1:
                    historical_slice = historical_slice.iloc[:-1]

                active_mask = ~historical_slice.iloc[-1].isna().to_numpy()
                live_cols = [col for idx, col in enumerate(self.asset_tickers) if active_mask[idx]]

                cleaned_slice = historical_slice[live_cols]
                vol_slice = self.volatility_panel[live_cols].loc[:current_date] if hasattr(self,
                                                                                           'volatility_panel') else None

                live_prev_weights = current_weights[name][active_mask].copy()
                if np.sum(live_prev_weights) > 0:
                    live_prev_weights = live_prev_weights / np.sum(live_prev_weights)
                else:
                    live_prev_weights = np.zeros(len(live_cols))

                try:
                    live_new_weights = strat.optimize_weights(cleaned_slice, live_prev_weights, vol_slice)
                    live_new_weights = np.nan_to_num(live_new_weights, nan=0.0)
                except Exception as e:
                    logging.error(f"Крах стратегии {name} на дату {current_date}: {e}")
                    live_new_weights = live_prev_weights

                new_global_weights = np.zeros(len(self.asset_tickers))
                global_live_indices = [self.asset_tickers.index(c) for c in live_cols]
                new_global_weights[global_live_indices] = live_new_weights

                current_weights[name] = new_global_weights
                weights_histories[name].iloc[t_idx] = current_weights[name]

        return weights_histories


class PortfolioBacktester:
    def __init__(self, board_panel: pd.DataFrame, inflation_annual: float = 0.075, commission: float = 0.0005):
        self.board_panel = board_panel.sort_index()
        self.daily_inflation = (1.0 + inflation_annual) ** (1.0 / 250.0) - 1.0
        self.commission = commission

    def _sim_core(self, weights_history: pd.DataFrame, initial_capital: float, extra_capital: float, scenario_type: str,
                  wherewithal: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        sim_dates = weights_history.index
        prices_subset = self.board_panel.loc[sim_dates]

        asset_tickers = [col for col in self.board_panel.columns if not col.endswith('_div')]
        div_tickers = [f"{ticker}_div" for ticker in asset_tickers]

        returns_matrix = prices_subset[asset_tickers].to_numpy()
        div_yield_matrix = prices_subset[div_tickers].to_numpy()
        weights_matrix = weights_history[asset_tickers].to_numpy()

        T = len(sim_dates)
        portfolio_values = np.zeros(T)
        benchmark_values = np.zeros(T)

        portfolio_values[0] = initial_capital
        benchmark_values[0] = initial_capital if scenario_type != "DCA" else extra_capital

        div_payout_queue = []
        dca_step_capital = extra_capital
        key_deposit = True

        for t in range(1, T):
            prev_capital = portfolio_values[t - 1]

            current_weights = weights_matrix[t - 1]
            prev_weights = weights_matrix[t - 2] if t > 1 else np.zeros(len(asset_tickers))

            turnover = np.sum(np.abs(current_weights - prev_weights))
            transaction_cost = prev_capital * turnover * self.commission
            active_capital = max(0.0, prev_capital - transaction_cost)

            day_returns = returns_matrix[t]
            clean_day_returns = np.nan_to_num(day_returns, nan=0.0)

            capital_growth = np.nansum(current_weights * clean_day_returns)
            active_capital *= (1.0 + capital_growth)

            day_div_yields = div_yield_matrix[t]
            clean_div_yields = np.nan_to_num(day_div_yields, nan=0.0)

            dividend_accrued = active_capital * np.nansum(current_weights * clean_div_yields)
            if dividend_accrued > 0:
                div_payout_queue.append((t + 15, dividend_accrued))

            dividend_payout = 0.0
            matured_divs = [item for item in div_payout_queue if item[0] <= t]
            div_payout_queue = [item for item in div_payout_queue if item[0] > t]
            for item in matured_divs:
                dividend_payout += item[1]

            active_capital += dividend_payout

            is_new_month = sim_dates[t].month != sim_dates[t - 1].month

            if scenario_type == "LUMPSUM":
                portfolio_values[t] = active_capital
                benchmark_values[t] = initial_capital * ((1.0 + self.daily_inflation) ** t)

            elif scenario_type == "DCA":
                dca_step_capital *= (1.0 + self.daily_inflation)
                month_replenishment = dca_step_capital if is_new_month else 0.0
                portfolio_values[t] = active_capital + month_replenishment
                benchmark_values[t] = benchmark_values[t - 1] + month_replenishment

            elif scenario_type == "FIRE":
                wherewithal *= (1.0 + self.daily_inflation)
                monthly_withdrawal = wherewithal if is_new_month else 0.0
                portfolio_values[t] = max(0.0, active_capital - monthly_withdrawal)

                if portfolio_values[t] <= 0:
                    portfolio_values[t:] = 0.0
                    break

                if key_deposit and benchmark_values[t - 1] - monthly_withdrawal > 0:
                    benchmark_values[t] = benchmark_values[t - 1] - monthly_withdrawal
                else:
                    benchmark_values[t] = 0.0
                    key_deposit = False

        return portfolio_values, benchmark_values

    def run_lumpsum_simulation(self, strategy_weights_history: pd.DataFrame, initial_capital: float = 1_000_000.0) -> pd.DataFrame:
        p_val, b_val = self._sim_core(strategy_weights_history, initial_capital, 0.0, "LUMPSUM")
        res = pd.DataFrame(index=strategy_weights_history.index)
        res['Nominal_Capital'] = p_val
        res['Inflation_Benchmark'] = b_val
        return res

    def run_dca_simulation(self, strategy_weights_history: pd.DataFrame, extra_capital: float = 50_000.0) -> pd.DataFrame:
        p_val, b_val = self._sim_core(strategy_weights_history, extra_capital, extra_capital, "DCA")
        res = pd.DataFrame(index=strategy_weights_history.index)
        res['Nominal_Capital'] = p_val
        res['Inflation_Benchmark'] = b_val
        return res

    def run_fire_simulation(self, strategy_weights_history: pd.DataFrame, initial_capital: float = 6_000_000.0, wherewithal: float = 60_000.0) -> pd.DataFrame:
        p_val, b_val = self._sim_core(strategy_weights_history, initial_capital, 0.0, "FIRE", wherewithal)
        res = pd.DataFrame(index=strategy_weights_history.index)
        res['Nominal_Capital'] = p_val
        res['Inflation_Benchmark'] = b_val
        return res[(res['Nominal_Capital'] > 0) | (res['Inflation_Benchmark'] > 0)]
