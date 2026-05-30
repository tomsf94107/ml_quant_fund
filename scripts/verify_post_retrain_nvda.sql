-- Run AFTER Pipeline B completes to verify insider_60d + multipliers OFF
-- Yesterday NVDA: prob_eff=0.708, BUY. After fix: expected lower, possibly HOLD.

SELECT prediction_date, horizon, signal,
       printf('%.4f', prob_up) AS eff,
       printf('%.4f', prob_raw) AS raw,
       printf('%.4f', prob_pct7) AS pct7,
       printf('%.3f', risk_mult) AS rm,
       printf('%.3f', regime_mult) AS gm,
       overlay_downgraded AS od,
       created_at
FROM predictions
WHERE ticker='NVDA' AND prediction_date >= date('now', '-1 days')
ORDER BY prediction_date DESC, horizon;

-- BUY signal distribution today
SELECT signal, COUNT(*) AS n,
       printf('%.3f', AVG(prob_up)) AS avg_eff,
       printf('%.3f', AVG(prob_raw)) AS avg_raw
FROM predictions
WHERE prediction_date = (SELECT MAX(prediction_date) FROM predictions)
GROUP BY signal;

-- Multipliers should average 1.0 if env var worked
SELECT printf('%.3f', AVG(risk_mult)) AS r,
       printf('%.3f', AVG(regime_mult)) AS reg
FROM predictions
WHERE prediction_date = (SELECT MAX(prediction_date) FROM predictions);
