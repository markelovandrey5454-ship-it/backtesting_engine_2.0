import numpy as np
import pandas as pd
from empty_box import BasePortfolioStrategy
import cvxpy as cp
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
from sklearn.covariance import ledoit_wolf

commission = 0.0005


class UniformStrategy(BasePortfolioStrategy):
    """Бенчмарк 1: Равновзвешенный инвестор.
    Делит капитал поровну между всеми доступными на момент ребалансировки активами."""
    def __init__(self):
        super().__init__(name="Uniform_1/N")

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        N = historical_returns.shape[1]
        weights = np.ones(N) / N
        return weights


class RandomMonkeyStrategy(BasePortfolioStrategy):
    """Бенчмарк 2: Случайный инвестор.
    Генерирует случайные веса на каждой ребалансировке. Ребалансировки реже обычного."""
    def __init__(self):
        super().__init__(name="Random_Monkey")

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        N = historical_returns.shape[1]
        random_vectors = np.random.rand(N)
        weights = random_vectors / np.sum(random_vectors)
        return weights


class StochasticMomentumStrategy(BasePortfolioStrategy):
    """Бенчмарк 3: Вероятностный Ротатор.
    Рассчитывает доходность активов за среднесрочное скользящее окно (3 месяца),
    отсекает отрицательные результаты, нормирует оставшиеся в вероятности
    и случайным образом выбирает, какие активы купить."""
    def __init__(self, lookback_days: int = 63, num_pick_assets: int = 3):
        super().__init__(name="Stochastic_Momentum")
        self.lookback = lookback_days
        self.num_picks = num_pick_assets

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        N = historical_returns.shape[1]
        weights = np.zeros(N)

        if len(historical_returns) < self.lookback:
            return np.ones(N) / N

        recent_history = historical_returns.tail(self.lookback)
        cum_returns = np.nanprod(1.0 + recent_history.to_numpy(), axis=0) - 1.0

        cum_returns[cum_returns < 0.0] = 0.0

        sum_positive = np.sum(cum_returns)

        if sum_positive == 0:
            return np.ones(N) / N

        probabilities = cum_returns / sum_positive
        probabilities = probabilities / np.sum(probabilities)

        chosen_indices = np.random.choice(N, size=min(self.num_picks, np.count_nonzero(cum_returns)), replace=False, p=probabilities)

        weights[chosen_indices] = 1.0 / len(chosen_indices)
        return weights / np.sum(weights)


class PersonalProfileStrategy(BasePortfolioStrategy):
    """Бенчмарк 4: Мой Инвестиционный Профиль.
    Распределяет капитал по жесткой иерархии классов активов: Возраст% в ОФЗ/Корпораты и возможно в ВДО,
    5% в Металлы по среднему распределению, Валютный коридор (переключение на Юань в 2022) и остаток в Акции."""
    def __init__(self, age_weight: float = 0.2, currency_weight: float = 0.10, lookback_days: int = 63):
        super().__init__(name="Personal_Profile")
        self.age_w = age_weight
        self.fx_w = currency_weight
        self.metals_w = 0.05
        self.reit_w = 0.03
        self.lookback = lookback_days

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        columns = list(historical_returns.columns)
        N = len(columns)
        weights = np.zeros(N)

        if len(historical_returns) < self.lookback:
            return np.ones(N) / N

        current_date = historical_returns.index[-1]
        recent_history = historical_returns.tail(self.lookback)

        equity_cols = [c for c in columns if c not in ['ОФЗ, фикс 1+', 'ОФЗ, фикс 5-10', 'ВДО, фикс', 'Денежный рынок(LQDT)',
                                                       'Доллар', 'Евро', 'Юань', 'Недвижимость', 'Золото', 'Серебро']]
        market_trend = 0.0
        if equity_cols:
            market_trend = np.nanmean(np.nanprod(1.0 + recent_history[equity_cols].to_numpy(), axis=0) - 1.0)

        if market_trend >= 0.10:
            current_cash_w = 0.10
        elif market_trend <= -0.10:
            current_cash_w = 0.0
        else:
            current_cash_w = 0.05 + (market_trend / 0.10) * 0.05

        if 'Денежный рынок(LQDT)' in columns:
            weights[columns.index('Денежный рынок(LQDT)')] = current_cash_w

            lqdt_ret = recent_history['Денежный рынок(LQDT)'].to_numpy()
            half = len(lqdt_ret) // 2
            rate_momentum = np.nansum(lqdt_ret[half:]) - np.nansum(lqdt_ret[:half])

            if rate_momentum > 0.0001:
                w_short, w_long, w_high_yield = 0.8, 0.1, 0.1
            elif rate_momentum < -0.0001:
                w_short, w_long, w_high_yield = 0.2, 0.7, 0.1
            elif lqdt_ret[-1] < (1.06) ** (1 / 250) - 1:
                w_short, w_long, w_high_yield = 0.3, 0.2, 0.5
            else:
                w_short, w_long, w_high_yield = 0.4, 0.6, 0.0

            if 'ОФЗ, фикс 1+' in columns:
                weights[columns.index('ОФЗ, фикс 1+')] = w_short * self.age_w
            if 'ОФЗ, фикс 5-10' in columns:
                weights[columns.index('ОФЗ, фикс 5-10')] = w_long * self.age_w
            if 'ВДО, фикс' in columns:
                weights[columns.index('ВДО, фикс')] = w_high_yield * self.age_w

        active_fx = 'Доллар' if current_date < pd.Timestamp('2022-06-01') else 'Юань'

        if active_fx in columns:
            global_fx_series = np.cumprod(1.0 + historical_returns[active_fx].to_numpy())
            fx_curr = global_fx_series[-1]
            fx_mean = np.mean(global_fx_series[-self.lookback:])

            if fx_curr <= fx_mean:
                fx_target_w = self.fx_w
            else:
                deviation = (fx_curr - fx_mean) / fx_mean
                fx_target_w = max(0.0, self.fx_w * (1.0 - deviation / 0.10))

            weights[columns.index(active_fx)] = fx_target_w

        if 'Золото' in columns and 'Серебро' in columns:
            idx_g = columns.index('Золото')
            idx_s = columns.index('Серебро')

            g_prices = np.cumprod(1.0 + recent_history['Золото'].to_numpy())
            s_prices = np.cumprod(1.0 + recent_history['Серебро'].to_numpy())

            met_curr = g_prices[-1] / s_prices[-1]

            l, m, h = 0.02, 0.03, 0.045
            if met_curr < 1:
                weights[idx_g] = min(h, m + (1.0 - met_curr) * (h - m) / 0.3)
            else:
                weights[idx_g] = max(l, m - (met_curr - 1) * (m - l) / 0.3)

            weights[idx_s] = self.metals_w - weights[idx_g]

        if 'Недвижимость' in columns:
            weights[columns.index('Недвижимость')] = self.reit_w

        allocated_cash = np.sum(weights)
        remaining_cash = 1.0 - allocated_cash

        if remaining_cash > 0 and equity_cols:
            equity_indices = [columns.index(c) for c in equity_cols]
            weights[equity_indices] = remaining_cash / len(equity_indices)

        return weights / np.sum(weights)


class MarkowitzStrategy(BasePortfolioStrategy):
    """Классическая Mean-Variance оптимизация Марковица в CVXPY.
        Минимизирует риск портфеля с учетом транзакционных издержек (L1-штраф)"""
    def __init__(self, target_daily_return: float = 0.0006):
        super().__init__(name="Markowitz_MV")
        self.target_return = target_daily_return
        self.commission = commission

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        live_cols = list(historical_returns.columns)
        N = len(live_cols)

        df_live = historical_returns[live_cols].tail(252)
        returns_np = df_live.to_numpy()

        mean_returns = np.nanmean(returns_np, axis=0)

        returns_clean = np.nan_to_num(returns_np, nan=0.0)
        sigma_live, _ = ledoit_wolf(returns_clean)

        w = cp.Variable(N)
        portfolio_risk = cp.quad_form(w, sigma_live)

        turnover_penalty = self.commission * cp.sum(cp.abs(w - prev_weights))

        objective = cp.Minimize(portfolio_risk + turnover_penalty)
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0
        ]

        if np.max(mean_returns) >= self.target_return:
            constraints.append(mean_returns @ w >= self.target_return)
        elif (median_returns := np.median(mean_returns)) > 0:
            constraints.append(mean_returns @ w >= median_returns)

        prob = cp.Problem(objective, constraints)

        prob.solve()
        return w.value


class MlHeavyweightStrategy(BasePortfolioStrategy):
    """Бенчмарк 6 (ИИ-Тяжеловес): Макро-Адаптивный Регрессионный Робот (SVR + Ridge).
    Обучается на импульсах цен, индивидуальной волатильности акций и ставках РЕПО.
    Распределяет капитал через экспоненциальный Softmax. В кризисы уходит на 100% в LQDT."""
    def __init__(self, train_window: int = 252, temperature: float = 0.05):
        super().__init__(name="ML_Macro_Heavyweight")
        self.train_window = train_window
        self.tau = temperature

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        columns = list(historical_returns.columns)
        N = len(columns)
        weights = np.zeros(N)

        if len(historical_returns) < self.train_window:
            return np.ones(N) / N

        df_train = historical_returns.tail(self.train_window).ffill().fillna(0.0)
        T_train = len(df_train)

        lqdt_ret = df_train['Денежный рынок(LQDT)'].to_numpy() if 'Денежный рынок(LQDT)' in columns else np.zeros(
            T_train)

        rate_ma = pd.Series(lqdt_ret).rolling(22, min_periods=1).mean().to_numpy()
        rate_change = rate_ma - lqdt_ret
        market_vol = df_train.std(axis=1).to_numpy()

        predicted_alphas = np.zeros(N)
        safe_names = ['ОФЗ, фикс 1+', 'ОФЗ, фикс 5-10', 'ВДО, фикс', 'Денежный рынок(LQDT)', 'Доллар', 'Евро', 'Юань',
                      'Недвижимость', 'Золото', 'Серебро']

        for i, col in enumerate(columns):
            if col in safe_names:
                continue

            asset_returns = df_train[col].to_numpy()

            mom_1m = pd.Series(asset_returns).rolling(22, min_periods=1).mean().to_numpy()
            mom_3m = pd.Series(asset_returns).rolling(63, min_periods=1).mean().to_numpy()
            asset_vol = pd.Series(asset_returns).rolling(22, min_periods=1).std().fillna(0.0).to_numpy()

            X = np.column_stack([mom_1m, mom_3m, asset_vol, rate_change, market_vol])

            X_clean = X[:-1]
            y_clean = asset_returns[1:]

            model_svr = LinearSVR(epsilon=0.0, C=1.0, loss='epsilon_insensitive', random_state=42, max_iter=2000)
            model_svr.fit(X_clean, y_clean)

            model_ridge = Ridge(alpha=1.0, random_state=42)
            model_ridge.fit(X_clean, y_clean)

            next_features = X[-1].reshape(1, -1)
            pred_svr = model_svr.predict(next_features)[0]
            pred_ridge = model_ridge.predict(next_features)[0]

            predicted_alphas[i] = 0.5 * pred_svr + 0.5 * pred_ridge

        predicted_alphas[predicted_alphas < 0.0] = 0.0
        sum_positive_signals = np.sum(predicted_alphas)

        if sum_positive_signals > 0.00001:
            exp_alphas = np.exp(predicted_alphas / self.tau)
            exp_alphas[predicted_alphas == 0.0] = 0.0

            equity_weights = exp_alphas / np.sum(exp_alphas)

            for i, col in enumerate(columns):
                if col not in safe_names:
                    weights[i] = equity_weights[i]

            remaining_cash = 1.0 - np.sum(weights)
            if remaining_cash > 0 and 'Денежный рынок(LQDT)' in columns:
                weights[columns.index('Денежный рынок(LQDT)')] = remaining_cash
        else:
            if 'Денежный рынок(LQDT)' in columns:
                weights[columns.index('Денежный рынок(LQDT)')] = 1.0
            else:
                weights = np.ones(N) / N

        return weights / np.sum(weights)


class RobustParabolicCvarStrategy_old(BasePortfolioStrategy):
    def __init__(self):
        super().__init__(name="High-risk prototype")
        self.commission = commission
        self.max_horizon = 1000
        self.current_market_panic = 1.0

    def _parabolic_integral(self, l_x, r_x, x_0, a, c):
        return (a / 3.0) * ((l_x - x_0) ** 3 - (r_x - x_0) ** 3) + c * (r_x - l_x)

    def _generate_parabolic_kernel_weights(self, returns_np: np.ndarray, vol_matrix_np: np.ndarray) -> np.ndarray:
        T = len(returns_np)
        W_total = np.zeros(T)

        market_track = np.mean(returns_np, axis=1)
        raw_market_vol = np.mean(vol_matrix_np, axis=1)
        t_axis = np.arange(T)
        convex_time_bonds = 1.0 - ((T - 1.0 - t_axis) / (T - 1.0)) ** 2
        vol_history = (raw_market_vol * convex_time_bonds) / (np.sum(convex_time_bonds) / T)
        current_vol = vol_history[-1]

        current_trend_val = np.nansum(market_track[-5:])
        if current_trend_val > 0.005: current_trend_regime = 1
        elif current_trend_val < -0.005: current_trend_regime = -1
        else: current_trend_regime = 0

        vol_std = np.std(vol_history)
        k_multiplier = 0.1
        similar_indices = []

        pilot_matches = 0
        for t in range(1, T):
            if abs(vol_history[t] - current_vol) <= vol_std: pilot_matches += 1
        min_target_proportions = max(3, pilot_matches // 3)

        while k_multiplier <= 3.0:
            similar_indices = []
            current_allowed_gap = k_multiplier * vol_std

            for t in range(1, T):
                past_trend_val = np.nansum(market_track[max(0, t - 5):t + 1])
                if past_trend_val > 0.005: past_trend_regime = 1
                elif past_trend_val < -0.005: past_trend_regime = -1
                else: past_trend_regime = 0

                if abs(vol_history[t] - current_vol) <= current_allowed_gap and past_trend_regime == current_trend_regime:
                    similar_indices.append(t)

            if len(similar_indices) >= min_target_proportions:
                break

            k_multiplier += 0.2

        count_similar = len(similar_indices)
        dynamic_base_hw = T / max(1, count_similar)
        dynamic_base_hw = np.clip(dynamic_base_hw, 20.0, 250.0)

        for t in similar_indices:
            x_0 = t
            h_w = dynamic_base_hw
            if x_0 - h_w < 0: continue

            a = 1.0 / (h_w ** 2)
            c = ((3.0 / 4.0) * np.sqrt(a)) ** (2.0 / 3.0)
            f = -a * (t_axis - x_0) ** 2 + c
            W_total += (f + np.abs(f)) / 2.0

        skew_window = 21
        skew_history = np.zeros(T)
        for t in range(skew_window, T):
            window = market_track[t - skew_window:t + 1]
            mean_w = np.mean(window)
            std_w = np.std(window)
            if std_w > 0: skew_history[t] = np.mean((window - mean_w) ** 3) / (std_w ** 3)

        working_skew = skew_history.copy()
        max_stretch_factor = 1.0
        crisis_kernels = []

        for _ in range(15):
            min_idx = np.argmin(working_skew)
            min_val = working_skew[min_idx]

            if min_val == 0.0:
                break

            x_0 = max(0, min_idx - skew_window)
            h_w = dynamic_base_hw
            a = 1.0 / (h_w ** 2)
            c = ((3.0 / 4.0) * np.sqrt(a)) ** (2.0 / 3.0)

            root_delta = np.sqrt(max(0.0, c / a))
            left_root = max(0, int(np.floor(x_0 - root_delta)))
            right_root = min(T - 1, int(np.ceil(x_0 + root_delta)))

            is_active_now = (T - 1 <= right_root)
            crisis_kernels.append((x_0, a, c, is_active_now))

            safe_left = min(left_root, max(0, min_idx - skew_window))
            safe_right = max(right_root, min(T - 1, min_idx + skew_window))
            working_skew[safe_left:safe_right + 1] = 0.0

        for x_0, a, c, is_active_now in crisis_kernels:
            f = -a * (t_axis - x_0) ** 2 + c
            positive_coupon = (f + np.abs(f)) / 2.0

            root_delta = np.sqrt(max(0.0, c / a))
            left_root = max(0, int(np.floor(x_0 - root_delta)))
            right_root = min(T - 1, int(np.ceil(x_0 + root_delta)))

            actual_area = self._parabolic_integral(left_root, right_root, x_0, a, c)

            stretch_factor = 1.0 / actual_area
            positive_coupon *= stretch_factor

            if is_active_now:
                if stretch_factor > max_stretch_factor:
                    max_stretch_factor = stretch_factor

            W_total += positive_coupon

        self.current_market_panic = max_stretch_factor
        return W_total / np.sum(W_total)

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        columns = list(historical_returns.columns)
        N = len(columns)

        df_window = historical_returns.tail(self.max_horizon)
        returns_np = df_window.to_numpy()
        market_mean_returns = np.nanmean(returns_np, axis=1)
        nan_indices_ret = np.isnan(returns_np)
        clean_returns_np = np.where(nan_indices_ret, market_mean_returns[:, None], returns_np)
        T_window = len(clean_returns_np)

        df_vol_window = volatility_history.tail(self.max_horizon)
        vol_matrix_np = df_vol_window.to_numpy()
        market_mean_vol = np.nanmean(vol_matrix_np, axis=1)
        nan_indices_vol = np.isnan(vol_matrix_np)
        clean_vol_matrix_np = np.where(nan_indices_vol, market_mean_vol[:, None], vol_matrix_np)

        W_days = self._generate_parabolic_kernel_weights(clean_returns_np, clean_vol_matrix_np)
        parabolic_mean_returns = np.sum(clean_returns_np * W_days[:, None], axis=0)

        beta = 0.95
        w = cp.Variable(N)
        alpha = cp.Variable()
        u = cp.Variable(T_window)

        old_losses = (-clean_returns_np / T_window) @ prev_weights
        old_alpha = np.percentile(old_losses, beta * 100)
        old_u = np.maximum(0, old_losses - old_alpha)
        old_cvar = old_alpha + (1.0 / (1.0 - beta)) * (W_days @ old_u)
        losses = (-clean_returns_np / T_window) @ w
        cvar_loss = alpha + (1.0 / (1.0 - beta)) * (W_days @ u)

        turnover_penalty = self.commission * cp.sum(cp.abs(w - prev_weights))

        old_growth = parabolic_mean_returns @ prev_weights
        growth_incentive = parabolic_mean_returns @ w

        objective = cp.Minimize(cvar_loss - old_cvar + turnover_penalty - growth_incentive + old_growth)

        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            u >= 0.0,
            u >= losses - alpha
        ]

        safe_names = ['Денежный рынок(LQDT)']
        semi_safe = ['Денежный рынок(LQDT)', "Юань", "Доллар", "Евро", "Недвижимость", "Золото"]

        for i, col in enumerate(columns):
            if col not in safe_names: constraints.append(w[i] <= 0.10)

        if self.current_market_panic >= 1.3:
            constraints.append(cp.sum([w[i] for i, col in enumerate(columns) if col in semi_safe]) >= min(1.0, self.current_market_panic / 2.0))

        prob = cp.Problem(objective, constraints)
        prob.solve()
        return w.value


class RobustParabolicCvarStrategy_new(BasePortfolioStrategy):
    def __init__(self, name: str = "Controlled-risk prototype"):
        super().__init__(name=name)
        self.commission = commission
        self.max_horizon = 1000
        self.target_area = 0.5
        self.current_market_panic = 1.0

    def _parabolic_integral(self, l_x, r_x, x_0, a, c):
        return (a / 3.0) * ((l_x - x_0) ** 3 - (r_x - x_0) ** 3) + c * (r_x - l_x)

    def _generate_parabolic_kernel_weights(self, returns_np: np.ndarray, vol_matrix_np: np.ndarray) -> np.ndarray:
        T = len(returns_np)
        W_total = np.zeros(T)
        t_axis = np.arange(T)

        mu_geom = np.exp(np.nanmean(np.log1p(returns_np), axis=0)) - 1.0
        raw_market_vol = np.mean(vol_matrix_np, axis=1)

        current_state = np.hstack([vol_matrix_np[-1, :], returns_np[-1, :]])
        rank_curr = np.argsort(np.argsort(current_state))

        spearman_scores = np.zeros(T - 1)
        for t in range(T - 1):
            past_state_t = np.hstack([vol_matrix_np[t, :], returns_np[t, :]])
            rank_past_t = np.argsort(np.argsort(past_state_t))

            corr_matrix = np.corrcoef(rank_curr, rank_past_t)
            spearman_scores[t] = corr_matrix[0, 1]

        similar_indices = [t for t, score in enumerate(spearman_scores) if score >= 0.6]
        if len(similar_indices) < 5: similar_indices = list(np.argsort(spearman_scores)[-5:])

        for t in similar_indices:
            vol_factor = raw_market_vol[t] / np.mean(raw_market_vol)
            h_w = int(20.0 / vol_factor)
            h_w = np.clip(h_w, 7.0, 37.0)
            x_0 = t + (h_w // 4)
            if x_0 - h_w // 2 < 0: continue
            c = 3.0 / (4.0 * h_w)
            a = 3.0 / (h_w ** 3)
            f = -a * (t_axis - x_0) ** 2 + c
            W_total += (f + np.abs(f)) / 2.0

        sigma_hist = np.nanstd(returns_np, axis=0)
        sigma_hist = np.where(sigma_hist == 0.0, 1e-8, sigma_hist)

        normalized_matrix = (returns_np - mu_geom) / sigma_hist
        working_skew = np.mean(normalized_matrix ** 3, axis=1)

        max_stretch_factor = 1.0
        crisis_kernels = []

        for _ in range(15):
            min_idx = np.argmin(working_skew)
            min_val = working_skew[min_idx]

            if min_val == 0.0:
                break

            x_0 = min_idx
            vol_factor = raw_market_vol[x_0] / np.mean(raw_market_vol)
            h_w = int(20.0 / vol_factor)
            h_w = np.clip(h_w, 3.0, 25.0)
            c = 3.0 / (4.0 * h_w)
            a = 3.0 / (h_w ** 3)

            root_delta = np.sqrt(max(0.0, c / a))
            left_root = max(0, int(np.floor(x_0 - root_delta)))
            right_root = min(T - 1, int(np.ceil(x_0 + root_delta)))

            is_active_now = (T - 1 == right_root)
            crisis_kernels.append((x_0, a, c, is_active_now))

            working_skew[left_root:(right_root + 1)] = 0.0

        for x_0, a, c, is_active_now in crisis_kernels:
            f = -a * (t_axis - x_0) ** 2 + c
            positive_coupon = (f + np.abs(f)) / 2.0

            root_delta = np.sqrt(max(0.0, c / a))
            left_root = max(0, int(np.floor(x_0 - root_delta)))
            right_root = min(T - 1, int(np.ceil(x_0 + root_delta)))

            actual_area = self._parabolic_integral(left_root, right_root, x_0, a, c)

            stretch_factor = self.target_area / actual_area
            positive_coupon *= stretch_factor

            if is_active_now:
                if stretch_factor > max_stretch_factor:
                    max_stretch_factor = stretch_factor

            W_total += positive_coupon

        self.current_market_panic = max_stretch_factor
        W_total[W_total == 0] += W_total[W_total > 0].min() / 2.0
        return W_total / np.sum(W_total)

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray,
                         volatility_history: pd.DataFrame = None) -> np.ndarray:
        columns = list(historical_returns.columns)
        N = len(columns)

        df_window = historical_returns.tail(self.max_horizon)
        returns_np = df_window.to_numpy()
        clean_returns_np = np.nan_to_num(returns_np, nan=0.0)
        T_window = len(clean_returns_np)

        df_vol_window = volatility_history.tail(self.max_horizon)
        vol_matrix_np = np.nan_to_num(df_vol_window.to_numpy(), nan=0.0005)

        W_days = self._generate_parabolic_kernel_weights(clean_returns_np, vol_matrix_np)
        parabolic_mean_returns = np.sum(clean_returns_np * W_days[:, None], axis=0)
        weighted_returns = clean_returns_np / T_window * self.current_market_panic

        beta = 0.99
        w = cp.Variable(N)
        alpha = cp.Variable()
        u = cp.Variable(T_window)

        old_losses = -weighted_returns @ prev_weights
        old_alpha = np.percentile(old_losses, beta * 100)
        old_u = np.maximum(0, old_losses - old_alpha)
        old_cvar = old_alpha + (1.0 / (1.0 - beta)) * np.mean(old_u)
        losses = -weighted_returns @ w
        cvar_loss = alpha + (1.0 / (1.0 - beta)) * cp.mean(u)

        turnover_penalty = self.commission * cp.sum(cp.abs(w - prev_weights))

        old_growth = parabolic_mean_returns @ prev_weights
        growth_incentive = parabolic_mean_returns @ w

        objective = cp.Minimize(cvar_loss - old_cvar + turnover_penalty - growth_incentive + old_growth)

        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            u >= 0.0,
            u >= losses - alpha
        ]

        safe_names = ['Денежный рынок(LQDT)']
        semi_safe = ['Денежный рынок(LQDT)', "Юань", "Доллар", "Евро", "Недвижимость", "Золото"]

        for i, col in enumerate(columns):
            if col not in safe_names: constraints.append(w[i] <= 0.10)

        if self.current_market_panic > 1.4:
            constraints.append(cp.sum([w[i] for i, col in enumerate(columns) if col in semi_safe]) >= 0.7)

        prob = cp.Problem(objective, constraints)
        prob.solve()
        return w.value
