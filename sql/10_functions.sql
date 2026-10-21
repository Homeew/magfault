-- =====================================================================
-- Functions and procedures. All are CREATE OR REPLACE (idempotent).
-- =====================================================================

-- Create the monthly partition of `readings` that contains p_ts, if missing.
CREATE OR REPLACE FUNCTION ensure_month_partition(p_ts timestamptz)
RETURNS text
LANGUAGE plpgsql AS $$
DECLARE
    m_start timestamptz := date_trunc('month', p_ts AT TIME ZONE 'UTC') AT TIME ZONE 'UTC';
    m_end   timestamptz := m_start + interval '1 month';
    part    text := format('readings_%s', to_char(m_start AT TIME ZONE 'UTC', 'YYYY_MM'));
BEGIN
    IF to_regclass(part) IS NULL THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF readings FOR VALUES FROM (%L) TO (%L)',
            part, m_start, m_end);
    END IF;
    RETURN part;
END $$;


-- Readings as the detectors see them under fault run p_run:
-- clean values, overridden wherever fault_samples has a row.
-- p_run = NULL returns the clean data.
CREATE OR REPLACE FUNCTION readings_observed(p_run integer)
RETURNS TABLE (iaga_code char(3), ts timestamptz,
               x double precision, y double precision,
               z double precision, f double precision)
LANGUAGE sql STABLE AS $$
    SELECT r.iaga_code, r.ts,
           CASE WHEN fs.ts IS NULL THEN r.x ELSE fs.x END,
           CASE WHEN fs.ts IS NULL THEN r.y ELSE fs.y END,
           CASE WHEN fs.ts IS NULL THEN r.z ELSE fs.z END,
           CASE WHEN fs.ts IS NULL THEN r.f ELSE fs.f END
    FROM readings r
    LEFT JOIN fault_samples fs
           ON fs.run_id = p_run
          AND fs.iaga_code = r.iaga_code
          AND fs.ts = r.ts
$$;


-- Minute of UTC day, 0..1439.
CREATE OR REPLACE FUNCTION utc_minute_of_day(p_ts timestamptz)
RETURNS integer
LANGUAGE sql IMMUTABLE AS $$
    SELECT (extract(hour FROM p_ts AT TIME ZONE 'UTC') * 60
          + extract(minute FROM p_ts AT TIME ZONE 'UTC'))::integer
$$;


-- ---------------------------------------------------------------------
-- build_features(run_id): the whole feature layer in SQL.
--
--  1. quiet days    : the p_quiet_days days per month with the lowest daily
--                     Kp sum (same idea as the IAGA international quiet days)
--  2. baseline      : per station, month and UTC minute-of-day, the median of
--                     x, y, z over quiet days, smoothed with a 31-min window
--  3. per minute    : d_f = |B| - F, disturbance dx/dy/dz/dh, |dB/dt|
--  4. robust scale  : median/MAD of dh and dbdt over quiet days -> z-scores
--  5. network       : mean z of the other stations, rolling corr with them
-- ---------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE build_features(p_run integer, p_quiet_days integer DEFAULT 5)
LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM features WHERE run_id = p_run;

    DROP TABLE IF EXISTS _obs, _quiet_days, _baseline, _f1, _scale, _f2;

    CREATE TEMP TABLE _obs AS
        SELECT o.*, (o.ts AT TIME ZONE 'UTC')::date AS day, utc_minute_of_day(o.ts) AS mod
        FROM readings_observed(p_run) o;
    CREATE INDEX ON _obs (iaga_code, ts);
    ANALYZE _obs;

    -- 1. quiet days ----------------------------------------------------
    CREATE TEMP TABLE _quiet_days AS
    WITH days AS (
        SELECT DISTINCT day FROM _obs
    ), day_kp AS (
        SELECT d.day, sum(k.kp) AS kp_sum
        FROM days d
        LEFT JOIN kp_index k
               ON k.ts_start >= d.day::timestamp AT TIME ZONE 'UTC'
              AND k.ts_start <  (d.day + 1)::timestamp AT TIME ZONE 'UTC'
        GROUP BY d.day
    ), ranked AS (
        SELECT day,
               row_number() OVER m AS rk,
               count(kp_sum) OVER (PARTITION BY date_trunc('month', day)) AS n_with_kp
        FROM day_kp
        WINDOW m AS (PARTITION BY date_trunc('month', day) ORDER BY kp_sum NULLS LAST, day)
    )
    SELECT day FROM ranked WHERE rk <= p_quiet_days OR n_with_kp = 0;   -- no Kp: every day counts

    -- 2. quiet-day baseline ---------------------------------------------
    CREATE TEMP TABLE _baseline AS
    WITH raw AS (
        SELECT o.iaga_code, date_trunc('month', o.day)::date AS month, o.mod,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY o.x) AS bx,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY o.y) AS by_,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY o.z) AS bz
        FROM _obs o
        JOIN _quiet_days q ON q.day = o.day
        GROUP BY 1, 2, 3
    )
    SELECT iaga_code, month, mod,
           avg(bx)  OVER w AS base_x,
           avg(by_) OVER w AS base_y,
           avg(bz)  OVER w AS base_z
    FROM raw
    WINDOW w AS (PARTITION BY iaga_code, month ORDER BY mod ROWS BETWEEN 15 PRECEDING AND 15 FOLLOWING);
    CREATE INDEX ON _baseline (iaga_code, month, mod);

    -- 3. per-minute quantities -----------------------------------------
    CREATE TEMP TABLE _f1 AS
    SELECT o.iaga_code, o.ts, o.day,
           sqrt(o.x * o.x + o.y * o.y + o.z * o.z) - o.f AS d_f,
           o.x - b.base_x AS dx,
           o.y - b.base_y AS dy,
           o.z - b.base_z AS dz,
           CASE WHEN o.ts - lag(o.ts) OVER w = interval '1 minute' THEN
                sqrt((o.x - lag(o.x) OVER w) ^ 2
                   + (o.y - lag(o.y) OVER w) ^ 2
                   + (o.z - lag(o.z) OVER w) ^ 2)
           END AS dbdt
    FROM _obs o
    LEFT JOIN _baseline b
           ON b.iaga_code = o.iaga_code
          AND b.month = date_trunc('month', o.day)::date
          AND b.mod = o.mod
    WINDOW w AS (PARTITION BY o.iaga_code ORDER BY o.ts);

    -- 4. robust per-station scale over quiet days ------------------------
    CREATE TEMP TABLE _scale AS
    WITH q AS (
        SELECT f.iaga_code, sqrt(f.dx ^ 2 + f.dy ^ 2) AS dh, f.dbdt
        FROM _f1 f JOIN _quiet_days qd ON qd.day = f.day
    ), med AS (
        SELECT iaga_code,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY dh)   AS med_dh,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY dbdt) AS med_dbdt
        FROM q GROUP BY iaga_code
    )
    SELECT m.iaga_code, m.med_dh, m.med_dbdt,
           greatest(1.4826 * percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(q.dh - m.med_dh)), 0.1)     AS s_dh,
           greatest(1.4826 * percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(q.dbdt - m.med_dbdt)), 0.05) AS s_dbdt
    FROM q JOIN med m USING (iaga_code)
    GROUP BY m.iaga_code, m.med_dh, m.med_dbdt;

    CREATE TEMP TABLE _f2 AS
    SELECT f.iaga_code, f.ts, f.d_f, f.dx, f.dy, f.dz,
           sqrt(f.dx ^ 2 + f.dy ^ 2) AS dh,
           f.dbdt,
           stddev_samp(f.d_f) OVER (PARTITION BY f.iaga_code ORDER BY f.ts
                                    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS d_f_std30,
           (sqrt(f.dx ^ 2 + f.dy ^ 2) - s.med_dh) / s.s_dh   AS z_dh,
           (f.dbdt - s.med_dbdt) / s.s_dbdt                 AS z_dbdt
    FROM _f1 f JOIN _scale s USING (iaga_code);
    CREATE INDEX ON _f2 (ts);

    -- 5. network context: leave-one-out means across stations at each minute
    INSERT INTO features (run_id, iaga_code, ts, d_f, d_f_std30, dx, dy, dz, dh, dbdt,
                          z_dh, z_dbdt, net_z_dh, net_z_dbdt, corr_dx_net, kp)
    WITH net AS (
        SELECT f.*,
               (sum(z_dh)   OVER t - coalesce(z_dh, 0))
                 / nullif(count(z_dh) OVER t - (z_dh IS NOT NULL)::int, 0)     AS net_z_dh,
               (sum(z_dbdt) OVER t - coalesce(z_dbdt, 0))
                 / nullif(count(z_dbdt) OVER t - (z_dbdt IS NOT NULL)::int, 0) AS net_z_dbdt,
               (sum(dx)     OVER t - coalesce(dx, 0))
                 / nullif(count(dx) OVER t - (dx IS NOT NULL)::int, 0)         AS net_dx
        FROM _f2 f
        WINDOW t AS (PARTITION BY ts)
    )
    SELECT p_run, n.iaga_code, n.ts, n.d_f, n.d_f_std30, n.dx, n.dy, n.dz, n.dh, n.dbdt,
           n.z_dh, n.z_dbdt, n.net_z_dh, n.net_z_dbdt,
           corr(n.dx, n.net_dx) OVER (PARTITION BY n.iaga_code ORDER BY n.ts
                                      ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),
           k.kp
    FROM net n
    LEFT JOIN kp_index k
           ON k.ts_start = date_bin('3 hours', n.ts, timestamptz '2000-01-01 00:00:00+00');

    DROP TABLE IF EXISTS _obs, _quiet_days, _baseline, _f1, _scale, _f2;
END $$;
