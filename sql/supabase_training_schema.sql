CREATE TABLE IF NOT EXISTS barangay_training_data (
    id           BIGSERIAL PRIMARY KEY,
    barangay_id  INTEGER NOT NULL,
    timestamp    TIMESTAMP,
    rainfall     DOUBLE PRECISION DEFAULT 0,
    humidity     DOUBLE PRECISION DEFAULT 0,
    soil         DOUBLE PRECISION DEFAULT 0,
    flood        DOUBLE PRECISION DEFAULT 0,
    storm_surge  DOUBLE PRECISION DEFAULT 0,
    risk_label   DOUBLE PRECISION NOT NULL
);

ALTER TABLE barangay_training_data
ADD COLUMN IF NOT EXISTS humidity DOUBLE PRECISION DEFAULT 0;

ALTER TABLE barangay_training_data
ADD COLUMN IF NOT EXISTS storm_surge DOUBLE PRECISION DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_training_barangay
ON barangay_training_data(barangay_id);

CREATE INDEX IF NOT EXISTS idx_training_barangay_timestamp
ON barangay_training_data(barangay_id, timestamp DESC);
