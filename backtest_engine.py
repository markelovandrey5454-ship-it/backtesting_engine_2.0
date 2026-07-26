import pandas as pd
import numpy as np
import logging


class PortfolioOrchestrator:
    def __init__(self, board_panel: pd.DataFrame, volatility_panel: pd.DataFrame, strategies: list, commission: float = 0.0005):
        self.board_panel = board_panel.sort_index()
        self.strategies = strategies
        self.commission = commission
        self.asset_tickers = [col for col in self.board_panel.columns if not col.endswith('_div')]
        self.volatility_panel = volatility_panel.sort_index()

    def generate_weights_history(self, start_date: str = "2015-01-01") -> dict[str, pd.DataFrame]:
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
                historical_slice = historical_slice.iloc[:-1]

                active_mask = ~historical_slice.iloc[-1].isna().to_numpy()
                live_cols = [col for idx, col in enumerate(self.asset_tickers) if active_mask[idx]]

                cleaned_slice = historical_slice[live_cols]
                vol_slice = self.volatility_panel[live_cols].loc[:current_date]
                vol_slice = vol_slice.iloc[:-1]

                div_cols = [f"{col}_div" for col in live_cols]
                div_slice = self.board_panel[div_cols].loc[:current_date].iloc[:-1]
                total_return_slice = cleaned_slice.to_numpy() + div_slice.to_numpy() * 0.87
                historical_returns_adjusted = pd.DataFrame(total_return_slice, index=cleaned_slice.index, columns=live_cols)

                last_day_returns = np.nan_to_num(historical_slice.iloc[-1].to_numpy(), nan=0.0)
                live_prev_weights = (current_weights[name] * (1.0 + last_day_returns))[active_mask].copy()
                if np.sum(live_prev_weights) > 0:
                    live_prev_weights = live_prev_weights / np.sum(live_prev_weights)
                else:
                    live_prev_weights = np.zeros(len(live_cols))

                try:
                    live_new_weights = strat.optimize_weights(historical_returns_adjusted, live_prev_weights, vol_slice)
                    live_new_weights = np.nan_to_num(live_new_weights, nan=0.0)
                except Exception as e:
                    logging.error(f"Крах стратегии {name} на дату {current_date}: {e}")
                    live_new_weights = live_prev_weights

                new_global_weights = np.zeros(len(self.asset_tickers))
                global_live_indices = [self.asset_tickers.index(c) for c in live_cols]
                new_global_weights[global_live_indices] = live_new_weights

                current_weights[name] = new_global_weights
                weights_histories[name].iloc[t_idx] = current_weights[name]

        #print(weights_histories["Robust_Parabolic_CVaR_Flagship"].iloc[-1])
        return weights_histories


class PortfolioBacktester:
    def __init__(self, board_panel: pd.DataFrame, inflation_annual: float = 0.075, commission: float = 0.0005):
        self.board_panel = board_panel.sort_index()
        self.daily_inflation = (1.0 + inflation_annual) ** (1.0 / 250.0) - 1.0
        self.commission = commission

    def _sim_core(self, weights_history: pd.DataFrame, initial_capital: float, extra_capital: float,
                  scenario_type: str, wherewithal: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
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
        prev_year_cap = initial_capital

        div_payout_queue = []
        key_deposit = True

        for t in range(1, T):
            prev_capital = portfolio_values[t - 1]

            if sim_dates[t].year != sim_dates[t - 1].year:
                tax_base = prev_capital - prev_year_cap
                if tax_base > 0:
                    tax_amount = tax_base * 0.13
                    prev_capital -= tax_amount
                prev_year_cap = prev_capital

            current_weights = weights_matrix[t]
            prev_weights = weights_matrix[t - 1]

            day_returns = returns_matrix[t]
            clean_day_returns = np.nan_to_num(day_returns, nan=0.0)

            capital_growth = np.sum(prev_weights * clean_day_returns)
            prev_capital *= (1.0 + capital_growth)

            day_div_yields = div_yield_matrix[t]

            dividend_accrued = prev_capital * np.nansum(prev_weights * day_div_yields)
            if dividend_accrued > 0:
                div_payout_queue.append((t + 15, dividend_accrued))

            dividend_payout = 0.0
            matured_divs = [item for item in div_payout_queue if item[0] <= t]
            div_payout_queue = [item for item in div_payout_queue if item[0] > t]
            for item in matured_divs:
                dividend_payout += item[1]

            prev_capital += dividend_payout

            prev_weights = (prev_weights * (1 + clean_day_returns))
            if (spv := np.sum(prev_weights)) > 0: prev_weights /= spv
            turnover = np.sum(np.abs(current_weights - prev_weights))
            transaction_cost = prev_capital * turnover * self.commission

            active_capital = max(0.0, prev_capital - transaction_cost)
            is_new_month = sim_dates[t].month != sim_dates[t - 1].month

            if scenario_type == "LUMPSUM":
                portfolio_values[t] = active_capital
                benchmark_values[t] = initial_capital * ((1.0 + self.daily_inflation) ** t)

            elif scenario_type == "DCA":
                extra_capital *= (1.0 + self.daily_inflation)
                month_replenishment = extra_capital if is_new_month else 0.0
                portfolio_values[t] = active_capital + month_replenishment
                benchmark_values[t] = benchmark_values[t - 1] * (1.0 + self.daily_inflation) + month_replenishment

            elif scenario_type == "FIRE":
                wherewithal *= (1.0 + self.daily_inflation)
                monthly_withdrawal = wherewithal if is_new_month else 0.0
                portfolio_values[t] = max(0.0, active_capital - monthly_withdrawal)

                if portfolio_values[t] == 0:
                    break

                if key_deposit and (benchmark_values[t - 1] * (1.0 + self.daily_inflation) - monthly_withdrawal > 0):
                    benchmark_values[t] = benchmark_values[t - 1] * (1.0 + self.daily_inflation) - monthly_withdrawal
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
