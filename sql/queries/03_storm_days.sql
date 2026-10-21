-- The most disturbed UTC days by daily max Kp, with each station's peak
-- horizontal disturbance and |dB/dt| on those days.
-- psql -v run=1 -f sql/queries/03_storm_days.sql

WITH daily_kp AS (
    SELECT (ts_start AT TIME ZONE 'UTC')::date AS day, max(kp) AS kp_max, sum(kp) AS kp_sum
    FROM kp_index
    GROUP BY 1
),
top_days AS (
    SELECT * FROM daily_kp ORDER BY kp_max DESC, kp_sum DESC LIMIT 5
)
SELECT t.day, t.kp_max, f.iaga_code,
       round(max(f.dh)::numeric, 1)   AS max_dh_nt,
       round(max(f.dbdt)::numeric, 1) AS max_dbdt_nt_per_min,
       round(avg(f.corr_dx_net)::numeric, 2) AS mean_corr_with_network
FROM top_days t
JOIN features f
  ON f.run_id = :run
 AND f.ts >= t.day::timestamp AT TIME ZONE 'UTC'
 AND f.ts <  (t.day + 1)::timestamp AT TIME ZONE 'UTC'
GROUP BY t.day, t.kp_max, f.iaga_code
ORDER BY t.kp_max DESC, t.day, f.iaga_code;
