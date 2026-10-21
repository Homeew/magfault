-- How coherent is the network? Correlation of the X disturbance between every
-- station pair, quiet (Kp < 3) versus storm (Kp >= 5) minutes.
-- Storm-time coherence drops with longitude separation (local-time effects),
-- which is exactly what limits the network residual test during storms.
-- psql -v run=1 -f sql/queries/04_station_pair_correlation.sql

SELECT a.iaga_code AS station_a, b.iaga_code AS station_b,
       round(abs(sa.longitude - sb.longitude)::numeric, 1)              AS dlon_deg,
       round(corr(a.dx, b.dx) FILTER (WHERE a.kp < 3)::numeric, 3)        AS corr_quiet,
       round(corr(a.dx, b.dx) FILTER (WHERE a.kp >= 5)::numeric, 3)       AS corr_storm,
       count(*) FILTER (WHERE a.kp >= 5)                                  AS storm_minutes
FROM features a
JOIN features b ON b.run_id = a.run_id AND b.ts = a.ts AND b.iaga_code > a.iaga_code
JOIN stations sa ON sa.iaga_code = a.iaga_code
JOIN stations sb ON sb.iaga_code = b.iaga_code
WHERE a.run_id = :run
GROUP BY 1, 2, 3
ORDER BY dlon_deg;
