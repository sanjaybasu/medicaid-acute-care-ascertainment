import sys, pathlib, warnings, numpy as np, pandas as pd
from datetime import date, timedelta
warnings.filterwarnings("ignore")
# Internal data connector (wm_conn) is loaded from the secure environment; not included here.
from wm_conn import coredb, query
cd = coredb("prod")
OUT='.'
T0 = sys.argv[1] if len(sys.argv)>1 else '2025-07-01'
t0 = date.fromisoformat(T0); BL=(t0-timedelta(days=365)).isoformat(); MID=(t0-timedelta(days=183)).isoformat(); FUP=(t0+timedelta(days=90)).isoformat()
print(f"T0={T0} BL={BL} MID={MID} FUP={FUP}")

cohort = query(cd, f"""
WITH spans AS (SELECT person_id, enrollment_start_date s, enrollment_end_date e FROM dbt_tuva_core.eligibility
   WHERE enrollment_start_date < DATE '{T0}' AND enrollment_end_date >= DATE '{BL}' AND enrollment_start_date > DATE '2015-01-01'),
bl AS (SELECT person_id, SUM(GREATEST(0,(LEAST(e,DATE '{T0}')-GREATEST(s,DATE '{BL}')))) bl_days FROM spans GROUP BY person_id),
et0 AS (SELECT DISTINCT person_id FROM dbt_tuva_core.eligibility WHERE enrollment_start_date<=DATE '{T0}' AND enrollment_end_date>=DATE '{T0}'),
pay AS (SELECT DISTINCT ON(person_id) person_id,payer,birth_date FROM dbt_tuva_core.eligibility ORDER BY person_id,enrollment_end_date DESC)
SELECT b.person_id,b.bl_days,p.payer,DATE_PART('year',AGE(DATE '{T0}',p.birth_date)) age
FROM bl b JOIN et0 USING(person_id) JOIN pay p USING(person_id) WHERE b.bl_days>=180""")
print("cohort:", f"{len(cohort):,}")

feat = query(cd, f"""
WITH enc AS (SELECT person_id, encounter_start_date d, encounter_group g, encounter_type t,
   attending_provider_id prov, facility_id fac, ed_flag, (encounter_group='inpatient') inpat
   FROM dbt_tuva_core.encounter WHERE data_source='Claims' AND encounter_start_date>=DATE '{BL}' AND encounter_start_date<DATE '{T0}'),
base AS (SELECT person_id,
   COUNT(*) FILTER (WHERE g='office based') n_office, COUNT(*) FILTER (WHERE t='urgent care') n_urgent,
   COUNT(*) FILTER (WHERE ed_flag=1) n_ed, COUNT(*) FILTER (WHERE inpat) n_inpat,
   COUNT(*) FILTER (WHERE g='outpatient') n_outp, COUNT(*) n_total,
   COUNT(DISTINCT fac) n_fac, COUNT(DISTINCT prov) n_prov, MAX(d) FILTER (WHERE g='office based') last_office,
   COUNT(*) FILTER (WHERE g='office based' AND d<DATE '{MID}') office_h1,
   COUNT(*) FILTER (WHERE g='office based' AND d>=DATE '{MID}') office_h2 FROM enc GROUP BY person_id),
provc AS (SELECT person_id,prov,COUNT(*) n FROM enc WHERE (prov IS NOT NULL AND g IN ('office based','outpatient')) OR t='urgent care' GROUP BY person_id,prov),
cont AS (SELECT person_id,SUM(n) N,SUM(n*n) sumsq,MAX(n) maxn,COUNT(*) nprov_amb FROM provc GROUP BY person_id)
SELECT b.*, c.N amb_visits,c.sumsq,c.maxn,c.nprov_amb,(DATE '{T0}'-b.last_office) days_since_office
FROM base b LEFT JOIN cont c USING(person_id)""")
print("features:", f"{len(feat):,}")

asc = query(cd, f"""
WITH lc AS (SELECT person_id,MAX(encounter_start_date) last_claim FROM dbt_tuva_core.encounter WHERE data_source='Claims' AND encounter_start_date<DATE '{T0}' GROUP BY person_id),
aa AS (SELECT DISTINCT person_id,1 adt_active FROM dbt_tuva_core.encounter WHERE data_source='HL7 ADT Feed' AND encounter_start_date>=DATE '{BL}' AND encounter_start_date<DATE '{T0}'),
g AS (SELECT person_id,COUNT(*) n_spans FROM dbt_tuva_core.eligibility WHERE enrollment_end_date>=DATE '{BL}' AND enrollment_start_date<DATE '{T0}' GROUP BY person_id)
SELECT l.person_id,(DATE '{T0}'-l.last_claim) claims_lag_days,COALESCE(a.adt_active,0) adt_active,COALESCE(g.n_spans,1) n_spans
FROM lc l LEFT JOIN aa a USING(person_id) LEFT JOIN g USING(person_id)""")
print("asc:", f"{len(asc):,}")

out = query(cd, f"""
WITH fup AS (SELECT person_id,
  MAX(CASE WHEN data_source='Claims' AND (ed_flag=1 OR encounter_group='inpatient') THEN 1 ELSE 0 END) y_claims,
  MAX(CASE WHEN data_source='HL7 ADT Feed' AND encounter_type IN ('emergency department','acute inpatient','observation') THEN 1 ELSE 0 END) y_adt,
  MAX(CASE WHEN data_source='HL7 ADT Feed' AND encounter_type IN ('emergency department','acute inpatient') THEN 1 ELSE 0 END) y_adt_noobs
  FROM dbt_tuva_core.encounter WHERE encounter_start_date>=DATE '{T0}' AND encounter_start_date<=DATE '{FUP}' GROUP BY person_id)
SELECT person_id,y_claims,y_adt,y_adt_noobs,GREATEST(y_claims,y_adt) y_union,GREATEST(y_claims,y_adt_noobs) y_union_noobs FROM fup""")
print("outcomes:", f"{len(out):,}")

df=cohort.merge(feat,on='person_id',how='left').merge(asc,on='person_id',how='left').merge(out,on='person_id',how='left')
for c in ['y_claims','y_adt','y_adt_noobs','y_union','y_union_noobs']: df[c]=df[c].fillna(0).astype(int)
p=f'{OUT}/dataset_{T0}.parquet'; df.to_parquet(p)
print("SAVED",p,"shape",df.shape,"| y_claims=%.4f y_union=%.4f y_union_noobs=%.4f"%(df.y_claims.mean(),df.y_union.mean(),df.y_union_noobs.mean()))
