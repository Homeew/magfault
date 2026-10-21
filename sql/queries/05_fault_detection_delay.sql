-- For every injected fault: first alarm of each method (LATERAL join) and the
-- detection delay. NULL delay = missed within the fault window + 20 min.
-- psql -v run=1 -f sql/queries/05_fault_detection_delay.sql

SELECT f.fault_id, f.iaga_code, f.fault_type, f.channel,
       round(f.magnitude::numeric, 4) AS magnitude,
       m.method,
       round((extract(epoch FROM first_alarm.ts - f.t_start) / 60)::numeric, 1) AS delay_min
FROM injected_faults f
CROSS JOIN (VALUES ('scalar_glrt'), ('network_glrt'), ('classifier'), ('iforest')) m(method)
LEFT JOIN LATERAL (
    SELECT d.ts
    FROM detections d
    WHERE d.run_id = f.run_id
      AND d.method = m.method
      AND d.iaga_code = f.iaga_code
      AND d.label = 1
      AND d.ts BETWEEN f.t_start AND f.t_end + interval '20 minutes'
    ORDER BY d.ts
    LIMIT 1
) first_alarm ON true
WHERE f.run_id = :run
ORDER BY f.fault_type, f.fault_id, m.method;
