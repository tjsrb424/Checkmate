# Raw Janggi Records

These files are original raw Janggi game records provided for the Oetongsu training pipeline. They are intentionally committed so another checkout can run the same conversion workflow.

| Stored file | Original file |
| --- | --- |
| `kakao-janggi-records-00.zip` | `카카오장기기보.zip` |
| `kakao-janggi-records-01.zip` | `카카오장기기보 (1).zip` |
| `kakao-janggi-records-02.zip` | `카카오장기기보 (2).zip` |
| `kakao-janggi-records-03.zip` | `카카오장기기보 (3).zip` |
| `amateur-master-records-01.zip` | `아마고수기보1.zip` |
| `amateur-master-records-02.zip` | `아마고수기보2.zip` |
| `amateur-master-records-05.zip` | `아마고수기보5.zip` |
| `amateur-master-records-06.zip` | `아마고수기보6.zip` |
| `janggi-note-cafe-tournament.gib` | `장기노트카페대회기보.gib` |
| `friendly-tournament-2-total.gib` | `제2회친선대회총보.gib` |

Default conversion:

```bash
npm run data:convert-raw-records
```

This writes `data/processed/janggi_clean_records.csv` and `data/processed/janggi_clean_records.conversion.json`.
