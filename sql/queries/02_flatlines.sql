-- Stuck-sensor candidates: runs of >= 10 identical consecutive values on any
-- vector channel, found with the gaps-and-islands trick on the observed data.
-- psql -v run=1 -f sql/queries/02_flatlines.sql

WITH obs AS (
    SELECT iaga_code, ts, ch, val
    FROM readings_observed(:run) o
    CROSS JOIN LATERAL (VALUES ('x', o.x), ('y', o.y), ('z', o.z)) v(ch, val)
    WHERE val IS NOT NULL
),
marked AS (
    SELECT *,
           CASE WHEN val = lag(val) OVER w THEN 0 ELSE 1 END AS new_run
    FROM obs
    WINDOW w AS (PARTITION BY iaga_code, ch ORDER BY ts)
),
islands AS (
    SELECT *, sum(new_run) OVER (PARTITION BY iaga_code, ch ORDER BY ts) AS run_no
    FROM marked
)
SELECT iaga_code, ch, min(ts) AS run_start, max(ts) AS run_end,
       count(*) AS minutes, min(val) AS value
FROM islands
GROUP BY iaga_code, ch, run_no
HAVING count(*) >= 10
ORDER BY minutes DESC
LIMIT 25;
