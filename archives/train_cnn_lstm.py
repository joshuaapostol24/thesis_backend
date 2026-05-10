from sqlalchemy import create_engine, text

# =====================================================
# SUPABASE CONNECTION
# =====================================================

DATABASE_URL = (
    "postgresql://postgres.jpovamcznyzoemcnjrgs:"
    "123Apostol%40Coco"
    "@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
)

engine = create_engine(DATABASE_URL)

# =====================================================
# CREATE BARANGAY HAZARD PROFILE
# =====================================================

query = """

DROP TABLE IF EXISTS barangay_hazard_profile;

CREATE TABLE barangay_hazard_profile AS

SELECT
    b.barangay_id,
    b.name AS barangay_name,
    b.lat,
    b.lon,

    -- =====================================================
    -- FLOOD HAZARD
    -- =====================================================

    CASE
        WHEN EXISTS (
            SELECT 1
            FROM mindoro_flood f
            WHERE
                f.flood_type = '25yr'
                AND ST_Intersects(
                    f.geometry,
                    ST_SetSRID(
                        ST_Point(b.lon, b.lat),
                        4326
                    )
                )
        )
        THEN 'High'

        WHEN EXISTS (
            SELECT 1
            FROM mindoro_flood f
            WHERE
                f.flood_type = '5yr'
                AND ST_Intersects(
                    f.geometry,
                    ST_SetSRID(
                        ST_Point(b.lon, b.lat),
                        4326
                    )
                )
        )
        THEN 'Medium'

        ELSE 'Low'
    END AS flood_hazard_level,

    CASE
        WHEN EXISTS (
            SELECT 1
            FROM mindoro_flood f
            WHERE
                f.flood_type = '25yr'
                AND ST_Intersects(
                    f.geometry,
                    ST_SetSRID(
                        ST_Point(b.lon, b.lat),
                        4326
                    )
                )
        )
        THEN 1.0

        WHEN EXISTS (
            SELECT 1
            FROM mindoro_flood f
            WHERE
                f.flood_type = '5yr'
                AND ST_Intersects(
                    f.geometry,
                    ST_SetSRID(
                        ST_Point(b.lon, b.lat),
                        4326
                    )
                )
        )
        THEN 0.6

        ELSE 0.2
    END AS flood_hazard_score,

    -- =====================================================
    -- STORM SURGE FLAGS
    -- =====================================================

    EXISTS (
        SELECT 1
        FROM storm_surge_zones s
        WHERE
            s.surge_type = 'SSA1'
            AND ST_Intersects(
                s.geometry,
                ST_SetSRID(
                    ST_Point(b.lon, b.lat),
                    4326
                )
            )
    ) AS in_ssa1,

    EXISTS (
        SELECT 1
        FROM storm_surge_zones s
        WHERE
            s.surge_type = 'SSA2'
            AND ST_Intersects(
                s.geometry,
                ST_SetSRID(
                    ST_Point(b.lon, b.lat),
                    4326
                )
            )
    ) AS in_ssa2,

    EXISTS (
        SELECT 1
        FROM storm_surge_zones s
        WHERE
            s.surge_type = 'SSA3'
            AND ST_Intersects(
                s.geometry,
                ST_SetSRID(
                    ST_Point(b.lon, b.lat),
                    4326
                )
            )
    ) AS in_ssa3,

    EXISTS (
        SELECT 1
        FROM storm_surge_zones s
        WHERE
            s.surge_type = 'SSA4'
            AND ST_Intersects(
                s.geometry,
                ST_SetSRID(
                    ST_Point(b.lon, b.lat),
                    4326
                )
            )
    ) AS in_ssa4,

    -- =====================================================
    -- STORM SURGE SCORE
    -- =====================================================

    CASE
        WHEN EXISTS (
            SELECT 1
            FROM storm_surge_zones s
            WHERE
                s.surge_type = 'SSA4'
                AND ST_Intersects(
                    s.geometry,
                    ST_SetSRID(
                        ST_Point(b.lon, b.lat),
                        4326
                    )
                )
        )
        THEN 1.0

        WHEN EXISTS (
            SELECT 1
            FROM storm_surge_zones s
            WHERE
                s.surge_type = 'SSA3'
                AND ST_Intersects(
                    s.geometry,
                    ST_SetSRID(
                        ST_Point(b.lon, b.lat),
                        4326
                    )
                )
        )
        THEN 0.8

        WHEN EXISTS (
            SELECT 1
            FROM storm_surge_zones s
            WHERE
                s.surge_type = 'SSA2'
                AND ST_Intersects(
                    s.geometry,
                    ST_SetSRID(
                        ST_Point(b.lon, b.lat),
                        4326
                    )
                )
        )
        THEN 0.5

        WHEN EXISTS (
            SELECT 1
            FROM storm_surge_zones s
            WHERE
                s.surge_type = 'SSA1'
                AND ST_Intersects(
                    s.geometry,
                    ST_SetSRID(
                        ST_Point(b.lon, b.lat),
                        4326
                    )
                )
        )
        THEN 0.3

        ELSE 0.1
    END AS storm_surge_score

FROM barangay_list b;

"""

# =====================================================
# EXECUTE QUERY
# =====================================================

with engine.begin() as conn:
    conn.execute(text(query))

print("barangay_hazard_profile created successfully!")