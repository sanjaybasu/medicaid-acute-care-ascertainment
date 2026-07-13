import sys, pathlib, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
# Internal data connector (wm_conn) loaded from secure environment; not included here.
from wm_conn import coredb, query
cd=coredb("prod")
BL,T0='2024-07-01','2025-07-01'
base=pd.read_parquet('dataset_2025-07-01.parquet'); pids=list(base.person_id.unique())
# enriched claims features over baseline
enr=query(cd, f"""
WITH enc AS (
 SELECT person_id, primary_diagnosis_code dx, paid_amount, encounter_start_date d
 FROM dbt_tuva_core.encounter
 WHERE data_source='Claims' AND encounter_start_date>=DATE '{BL}' AND encounter_start_date<DATE '{T0}'
   AND person_id = ANY(%(p)s))
SELECT person_id,
  COUNT(DISTINCT LEFT(dx,3)) FILTER (WHERE dx IS NOT NULL) n_dx_cat,
  SUM(paid_amount) tot_paid,
  COUNT(*) FILTER (WHERE dx LIKE 'E10%%' OR dx LIKE 'E11%%') dx_diabetes,
  COUNT(*) FILTER (WHERE dx LIKE 'I50%%') dx_chf,
  COUNT(*) FILTER (WHERE dx LIKE 'J4%%') dx_resp,
  COUNT(*) FILTER (WHERE dx LIKE 'N18%%') dx_ckd,
  COUNT(*) FILTER (WHERE dx LIKE 'F%%') dx_bh
FROM enc GROUP BY person_id""", p=pids)
ph=query(cd, f"""SELECT person_id, COUNT(*) n_rx, COUNT(DISTINCT ndc_code) n_rx_distinct
 FROM dbt_tuva_core.pharmacy_claim WHERE dispensing_date>=DATE '{BL}' AND dispensing_date<DATE '{T0}'
   AND person_id = ANY(%(p)s) GROUP BY person_id""", p=pids)
enr=enr.merge(ph,on='person_id',how='left')
enr.to_parquet('_enriched_features.parquet'); print("enriched features:", enr.shape)

# event sequences (claims+ADT), ordered, for the sequence model
seq=query(cd, f"""
SELECT person_id, encounter_start_date d,
  CASE WHEN data_source='HL7 ADT Feed' THEN 'adt_'||encounter_type ELSE encounter_group END etype
FROM dbt_tuva_core.encounter
WHERE encounter_start_date>=DATE '{BL}' AND encounter_start_date<DATE '{T0}' AND person_id = ANY(%(p)s)
""", p=pids)
seq['d']=pd.to_datetime(seq.d); t0=pd.Timestamp(T0)
seq['wk']=((t0-seq.d).dt.days//7).clip(0,55).astype(int)
seq=seq.sort_values(['person_id','d'])
seq.to_parquet('_event_sequences.parquet'); print("sequence rows:", len(seq), "distinct etypes:", seq.etype.nunique())
