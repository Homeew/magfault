-- =====================================================================
-- magfault schema
--   raw layer      : stations, ingest_files, readings (monthly partitions), kp_index
--   ground truth   : fault_runs, injected_faults, fault_samples
--   analysis layer : features, detections
-- Every statement is idempotent so `magfault initdb` can be re-run safely.
-- =====================================================================

-- New sessions on this database display timestamps in UTC.
DO $$ BEGIN
    EXECUTE format('ALTER DATABASE %I SET timezone TO %L', current_database(), 'UTC');
END $$;

CREATE TABLE IF NOT EXISTS stations (
    iaga_code     char(3) PRIMARY KEY,
    name          text,
    latitude      double precision,
    longitude     double precision,
    elevation_m   double precision,
    reported      text,             -- element set as published, e.g. XYZF, HDZF
    source        text              -- "INTERMAGNET" or "synthetic"
);

-- One row per raw file loaded. sha256 makes reloads of the same file a no-op.
CREATE TABLE IF NOT EXISTS ingest_files (
    ingest_id     serial PRIMARY KEY,
    iaga_code     char(3) NOT NULL REFERENCES stations,
    file_name     text NOT NULL,
    sha256        char(64) NOT NULL UNIQUE,
    first_ts      timestamptz,
    last_ts       timestamptz,
    n_rows        integer,
    n_missing     integer,
    loaded_at     timestamptz NOT NULL DEFAULT now()
);

-- 1-minute vector (x, y, z) and independent scalar (f) field, nT.
-- Partitioned by month; partitions are created on demand by ensure_month_partition().
CREATE TABLE IF NOT EXISTS readings (
    iaga_code  char(3) NOT NULL REFERENCES stations,
    ts         timestamptz NOT NULL,
    x          double precision,
    y          double precision,
    z          double precision,
    f          double precision,
    ingest_id  integer,
    PRIMARY KEY (iaga_code, ts)
) PARTITION BY RANGE (ts);

CREATE INDEX IF NOT EXISTS readings_ts_idx ON readings (ts);

-- Planetary 3-hour Kp (GFZ Potsdam). ts_start is the start of the 3-hour interval.
CREATE TABLE IF NOT EXISTS kp_index (
    ts_start  timestamptz PRIMARY KEY,
    kp        real NOT NULL CHECK (kp BETWEEN 0 AND 9),
    status    text
);

-- ---------------------------------------------------------------------
-- Ground truth: each fault run is one reproducible corruption of the
-- clean readings. The clean table is never modified; fault_samples holds
-- override values that readings_observed(run_id) overlays at query time.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fault_runs (
    run_id           serial PRIMARY KEY,
    seed             integer NOT NULL,
    calibration_end  timestamptz NOT NULL,   -- no faults are injected before this
    params           jsonb NOT NULL DEFAULT '{}',
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS injected_faults (
    fault_id    serial PRIMARY KEY,
    run_id      integer NOT NULL REFERENCES fault_runs ON DELETE CASCADE,
    iaga_code   char(3) NOT NULL REFERENCES stations,
    fault_type  text NOT NULL,
    channel     text NOT NULL,               -- x | y | z | f | xyz
    t_start     timestamptz NOT NULL,
    t_end       timestamptz NOT NULL,        -- inclusive
    magnitude   double precision NOT NULL,
    params      jsonb NOT NULL DEFAULT '{}',
    CHECK (t_end >= t_start)
);
CREATE INDEX IF NOT EXISTS injected_faults_run_idx ON injected_faults (run_id, iaga_code, t_start);

CREATE TABLE IF NOT EXISTS fault_samples (
    run_id     integer NOT NULL REFERENCES fault_runs ON DELETE CASCADE,
    iaga_code  char(3) NOT NULL,
    ts         timestamptz NOT NULL,
    x          double precision,
    y          double precision,
    z          double precision,
    f          double precision,
    fault_id   integer NOT NULL REFERENCES injected_faults ON DELETE CASCADE,
    PRIMARY KEY (run_id, iaga_code, ts)
);

-- ---------------------------------------------------------------------
-- Feature table written by build_features(run_id).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    run_id        integer NOT NULL REFERENCES fault_runs ON DELETE CASCADE,
    iaga_code     char(3) NOT NULL,
    ts            timestamptz NOT NULL,
    d_f           double precision,  -- |B_vector| - F_scalar  (vector/scalar consistency residual)
    d_f_std30     double precision,  -- 30-min rolling std of d_f
    dx            double precision,  -- disturbance relative to quiet-day baseline
    dy            double precision,
    dz            double precision,
    dh            double precision,  -- horizontal disturbance magnitude
    dbdt          double precision,  -- |dB/dt|, nT/min
    z_dh          double precision,  -- robust z-score of dh (station scale from quiet days)
    z_dbdt        double precision,
    net_z_dh      double precision,  -- mean z_dh of the OTHER stations at the same minute
    net_z_dbdt    double precision,
    corr_dx_net   double precision,  -- 60-min rolling corr(dx, network-mean dx excluding self)
    kp            real,
    PRIMARY KEY (run_id, iaga_code, ts)
);

-- Alarm minutes only (sparse). label: 1 = sensor fault, 2 = geomagnetic event.
CREATE TABLE IF NOT EXISTS detections (
    run_id     integer NOT NULL REFERENCES fault_runs ON DELETE CASCADE,
    method     text NOT NULL,
    iaga_code  char(3) NOT NULL,
    ts         timestamptz NOT NULL,
    score      double precision,
    label      smallint NOT NULL DEFAULT 1 CHECK (label IN (1, 2)),
    PRIMARY KEY (run_id, method, iaga_code, ts)
);
