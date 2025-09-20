"""simple_env.py
Fallback lightweight multi-asset portfolio environment when TensorTrade is unavailable.

Features:
- Accepts preloaded price dataframe with columns: symbol, datetime, close, optional engineered features.
- Action: target raw weights per symbol (continuous Box). We l1-normalize to sum(|w|) 1:
            self.returns[:-1] = (self.closes[1:] - self.closes[:-1]) / (self.closes[:-1] + 1e-12)
        self.n_steps, self.n_symbols = self.closes.shape
        self.n_feat_per_symbol = len(config.feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_symbols * self.n_feat_per_symbol,), dtype=np.float32
        )
        w_low = -1.0 if config.allow_shorts else 0.0
        self.action_space = spaces.Box(low=w_low, high=1.0, shape=(self.n_symbols,), dtype=np.float32)
        self._step_idx = None
        self._weights = np.zeros(self.n_symbols, dtype=np.float32)

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        self._step_idx = self.cfg.window
        if self._step_idx >= self.n_steps:
            self._step_idx = 0
        self._weights[:] = 0.0
        obs = self._current_obs()
        return obs, {}

    def _current_obs(self):
        if self._step_idx >= self.n_steps:
            idx = self.n_steps - 1
        else:
            idx = self._step_idx
        f_slice = self.features[idx]
        return f_slice.astype(np.float32)

    def step(self, action):  # type: ignore[override]
        action = np.asarray(action, dtype=np.float32)
        if np.sum(np.abs(action)) > 1e-9:
            norm = np.sum(np.abs(action))
            action = action / max(1.0, norm)
        turnover = np.mean(np.abs(action - self._weights))
        done = False
        if self._step_idx >= self.n_steps - 2:
            done = True
        r_vec = self.returns[self._step_idx + 1] if not done else self.returns[self._step_idx]
        portfolio_ret = float(np.dot(action, r_vec))
        reward = portfolio_ret - self.cfg.turnover_penalty * turnover
        self._weights = action
        self._step_idx += 1
        obs = self._current_obs()
        truncated = False
        info = {"portfolio_return": portfolio_ret, "turnover": turnover}
        return obs, reward, done, truncated, info

    def render(self):  # pragma: no cover
        pass
