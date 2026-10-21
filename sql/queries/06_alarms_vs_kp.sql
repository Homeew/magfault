-- Alarm-minute rate per method as a function of geomagnetic activity.
-- A sensor-fault detector should be flat in Kp; one that rises with Kp is
-- confusing space weather with instrument problems.
-- psql -v run=1 -f sql/queries/06_alarms_vs_kp.sql

WITH minutes AS (
    SELECT f.iaga_code, f.ts, floor(f.kp)::int AS kp_bin
    FROM features f
    JOIN fault_runs r ON r.run_id = f.run_id
    WHERE f.run_id = :run
      AND f.ts >= r.calibration_end
      AND NOT EXISTS (                          -- exclude true fault minutes
          SELECT 1 FROM injected_faults i
          WHERE i.run_id = f.run_id AND i.iaga_code = f.iaga_code
            AND f.ts BETWEEN i.t_start AND i.t_end + interval '20 minutes')
)
SELECT m.kp_bin,
       count(*) AS minutes,
       meth.method,
       round(1e3 * count(d.ts)::numeric / count(*), 3) AS alarms_per_1000_minutes
FROM minutes m
CROSS JOIN (SELECT DISTINCT method FROM detections WHERE run_id = :run) meth
LEFT JOIN detections d
       ON d.run_id = :run AND d.method = meth.method
      AND d.iaga_code = m.iaga_code AND d.ts = m.ts AND d.label = 1
GROUP BY m.kp_bin, meth.method
ORDER BY meth.method, m.kp_bin;
