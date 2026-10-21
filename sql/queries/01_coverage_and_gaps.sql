-- Data coverage per station and month, plus the longest gap (gaps-and-islands).
-- psql -f sql/queries/01_coverage_and_gaps.sql

WITH monthly AS (
    SELECT iaga_code,
           date_trunc('month', ts) AS month,
           count(*)                                          AS rows,
           count(*) FILTER (WHERE x IS NOT NULL AND f IS NOT NULL) AS complete
    FROM readings
    GROUP BY 1, 2
),
present AS (                     -- minutes with a usable vector sample
    SELECT iaga_code, ts,
           ts - lag(ts) OVER (PARTITION BY iaga_code ORDER BY ts) AS step
    FROM readings
    WHERE x IS NOT NULL AND y IS NOT NULL AND z IS NOT NULL
),
gaps AS (
    SELECT iaga_code, date_trunc('month', ts) AS month,
           max(step) - interval '1 minute' AS longest_gap
    FROM present
    GROUP BY 1, 2
)
SELECT m.iaga_code,
       to_char(m.month, 'YYYY-MM')                                  AS month,
       m.rows,
       round(100.0 * m.complete / m.rows, 2)                        AS pct_complete,
       g.longest_gap
FROM monthly m
LEFT JOIN gaps g USING (iaga_code, month)
ORDER BY m.iaga_code, m.month;
