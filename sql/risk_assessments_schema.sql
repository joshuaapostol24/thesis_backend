CREATE TABLE IF NOT EXISTS risk_assessments (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    barangay_id     INTEGER NOT NULL,
    hazard          TEXT NOT NULL,
    location        TEXT,
    rainfall        DOUBLE PRECISION,
    humidity        DOUBLE PRECISION,
    soil            DOUBLE PRECISION,
    flood           DOUBLE PRECISION,
    storm_surge     DOUBLE PRECISION,
    rule_score      DOUBLE PRECISION,
    predicted       DOUBLE PRECISION,
    final_risk      DOUBLE PRECISION,
    risk_level      TEXT NOT NULL,
    osm_is_fallback BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_risk_assessments_barangay
ON risk_assessments(barangay_id);

CREATE INDEX IF NOT EXISTS idx_risk_assessments_timestamp
ON risk_assessments(timestamp DESC);